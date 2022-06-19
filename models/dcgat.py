import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn.norm import GraphNorm
# from torch_geometric.nn import GATConv

import os
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.manifold import TSNE
from utils.training_utils import plot_tsne_reduced_embeddings

device = 'cuda' if torch.cuda.is_available() else 'cpu'



from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from torch_geometric.nn.inits import glorot, zeros


class GATConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    If the graph has multi-dimensional edge features :math:`\mathbf{e}_{i,j}`,
    the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j
        \, \Vert \, \mathbf{\Theta}_{e} \mathbf{e}_{i,j}]\right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k
        \, \Vert \, \mathbf{\Theta}_{e} \mathbf{e}_{i,k}]\right)\right)}.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). (default: :obj:`None`)
        fill_value (float or Tensor or str, optional): The way to generate
            edge features of self-loops (in case :obj:`edge_dim != None`).
            If given as :obj:`float` or :class:`torch.Tensor`, edge features of
            self-loops will be directly given by :obj:`fill_value`.
            If given as :obj:`str`, edge features of self-loops are computed by
            aggregating all features of edges that point to the specific node,
            according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`"mean"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value

        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        if isinstance(in_channels, int):
            self.lin_src = Linear(in_channels, heads * out_channels,
                                  bias=False, weight_initializer='glorot')
            self.lin_dst = self.lin_src
        else:
            self.lin_src = Linear(in_channels[0], heads * out_channels, False,
                                  weight_initializer='glorot')
            self.lin_dst = Linear(in_channels[1], heads * out_channels, False,
                                  weight_initializer='glorot')

        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
            self.att_edge = Parameter(torch.Tensor(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        glorot(self.att_edge)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None,
                return_attention_weights=None):
        # type: (Union[Tensor, OptPairTensor], Tensor, OptTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, OptTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], Tensor, OptTensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, OptTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = x_dst = self.lin_src(x).view(-1, H, C)
        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor, edge_attr: OptTensor)  # noqa
        out = self.propagate(edge_index, x=x, alpha=alpha, edge_attr=edge_attr,
                             size=size)

        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha  # Save for later use.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)





class DCGAT(torch.nn.Module):
    """
    Architecture expects to receive 1 day graphs, so that adjacency matrix can change every day. N 
    1-day graphs passed through, then take model output for N+1 day. 

    Constructor Arguments:
        node_features: number of features each node in 1-day graph contains
        emd_dim: embedding dimension that GAT layers will output
        window_size: number of 1-day graphs that will be passed through network before predicting next day
        output: number of features to predict for each node on N+1 day
        training: whether the model is being loaded for training or test. Affects things like dropout if present.
        lstm_type: what type of LSTM to use (["vanila"])
    """

    def __init__(self,
                node_features: int = 1, 
                emb_dim: int = 10,
                window_size: int = 7,
                output: int = 1, 
                name: str = "DCGAT"):
        super(DCGAT, self).__init__()

        self.emb_dim = emb_dim
        self.window_size = window_size
        self.name = name

        self.gat1 = GATConv(in_channels=node_features, out_channels=self.emb_dim // 5, heads=5, edge_dim=1)
        self.gat2 = GATConv(in_channels=self.emb_dim, out_channels=self.emb_dim // 5, heads=5, edge_dim=1)

        self.graph_norm_1 = GraphNorm(self.emb_dim)
        self.graph_norm_2 = GraphNorm(self.emb_dim)

        self.lstm1 = nn.LSTMCell(input_size=2 * self.emb_dim, hidden_size=self.emb_dim)
        self.lstm2 = nn.LSTMCell(input_size=self.emb_dim, hidden_size=self.emb_dim)
        
        self.act1 = torch.nn.ReLU()
        self.lin1 = torch.nn.Linear(self.window_size + (2 * self.emb_dim), output)
        
        self.concat_feat_list = []

    def forward(self, data: Data, h_1: Tensor=None, c_1: Tensor=None, 
                h_2: Tensor=None, c_2: Tensor=None, day_idx: int=0):
        # Get data from snapshot
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        """
        x is [10, 3]
        edge_index is [2, num_edges]
        edge_attr is [num_edges, 1]
        last_day_flag is False unless day graph being passed is last day graph in the time window
        """
        
        graph_outputs = []
        self.concat_feat_list.append(x)  # [:,1:2]

        # First GNN Layer
        x, (alphas_edge_index1, attention_weights1) = self.gat1(x, edge_index=edge_index, edge_attr=edge_attr, return_attention_weights=True)
        x = self.graph_norm_1(x)
        x = F.relu(x)
        graph_outputs.append(x)
        
        x, (alphas_edge_index2, attention_weights2) = self.gat2(x, edge_index=edge_index, edge_attr=edge_attr, return_attention_weights=True)
        x = self.graph_norm_2(x)
        x = F.relu(x)
        graph_outputs.append(x)

        x = torch.cat(graph_outputs, dim=1)

        # Initialize hidden and cell states if None
        if h_1 is None:
            h_1 = torch.zeros(x.shape[0], self.emb_dim)
        if c_1 is None:
            c_1 = torch.zeros(x.shape[0], self.emb_dim)
        if h_2 is None:
            h_2 = torch.zeros(x.shape[0], self.emb_dim)
        if c_2 is None:
            c_2 = torch.zeros(x.shape[0], self.emb_dim)

        # RNN Layer
        h_1, c_1 = self.lstm1(x, (h_1, c_1))  # h_1 and c_1 both become [10, self.emb_dim]
        h_2, c_2 = self.lstm2(h_1, (h_2, c_2))  # h_2 and c_2 both become [10, self.emb_dim]
        
        if day_idx == self.window_size - 1:
            # Skip connection
            concat_feat = torch.cat(self.concat_feat_list, dim=1)  # becomes [10, 16]
            x = torch.cat((concat_feat, h_1, h_2), dim=1)  # x becomes [10, 16 + 2 * self.emb_dim]
            self.concat_feat_list.clear()

            x = self.act1(x)
            x = self.lin1(x)

        return x, h_1, c_1, h_2, c_2, (alphas_edge_index1, attention_weights1, alphas_edge_index2, attention_weights2)


    def visualize_gradients(self, split_name, save_path, epoch_idx, color="C0"):
        """
        Visualization code partly taken from the amazing notebook tutorial at
        https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial3/Activation_Functions.html
        """
        # We limit our visualization to the gradients of only weight parameters and exclude the bias to reduce the number of plots
        grads = {name: params.grad.data.view(-1) for name, params in list(self.named_parameters()) if "weight" in name and params.grad is not None}  # Only seeing gradients for leaf nodes, non-leaves grad is None

        # Plot activation distributions
        gradient_distrib_save_path = os.path.join(save_path, "gradient_distrib_plots")
        if not os.path.exists(gradient_distrib_save_path):
            os.mkdir(gradient_distrib_save_path)
        
        columns = len(grads)
        fig, ax = plt.subplots(1, columns, figsize=(columns*4, 4))
        fig_index = 0
        for key in grads:
            key_ax = ax[fig_index%columns]
            sns.histplot(data=grads[key], bins=30, ax=key_ax, color=color, kde=True)
            mean = grads[key].mean()
            median = grads[key].median()
            mode = grads[key].flatten().mode(dim=0)[0].item()
            std = grads[key].std()
            key_ax.set_title(str(key) + "\nMean: {:.4f}, Median: {:.4f}\nMode: {:.4f}, STD: {:.4f}".format(mean, median, mode, std))
            key_ax.set_xlabel("Grad magnitude")
            fig_index += 1
        fig.suptitle(f"Gradient Magnitude Distribution on Full " + split_name + " Dataset", fontsize=14, y=1.05)
        fig.subplots_adjust(wspace=0.45)
        # plt.show()
        plt.tight_layout()
        plt.savefig(os.path.join(gradient_distrib_save_path, split_name + "_gradient_distrib_epoch_" + str(epoch_idx)), bbox_inches='tight')
        plt.clf()
        plt.close()
    

    def plot_grad_flow(self, save_path, epoch_idx):
        """
        Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.
        
        Usage: Plug this function in Trainer class after loss.backwards() as 
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
        """
        gradient_flow_plot_save_path = os.path.join(save_path, "gradient_flow_plots")
        if not os.path.exists(gradient_flow_plot_save_path):
            os.mkdir(gradient_flow_plot_save_path)

        ave_grads = []
        max_grads= []
        layers = []
        for n, p in self.named_parameters():
            # if (p.requires_grad) and ("bias" not in n):
            if "weight" in n and p.grad is not None:
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
        plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
        plt.savefig(os.path.join(gradient_flow_plot_save_path, "gradient_flow_plot_ep{}".format(epoch_idx)), bbox_inches='tight', facecolor="white")
        plt.clf()
        plt.close()


    def visualize_spatial_embeddings(self, args, data_loader, split_name, save_path, epoch_idx):
        gat1_emb_list = []
        gat2_emb_list = []
        gat_concat_emb_list = []
        self.eval()

        # Iterate over each day in dataset. This loop doesn't windows, just plain days in dataset
        for day_idx, (day_node_feat, day_edge_idx, day_edge_attr) in enumerate(zip(data_loader.dataset.orig_feature_matrix, data_loader.dataset.orig_edge_idx, data_loader.dataset.orig_edge_weights)):
            if args["num_node_features"] == 1:
                day_node_feat = day_node_feat[:, 2:3]
            
            cutoff_idx = day_edge_idx[0].tolist().index(-1)
            day_edge_idx = day_edge_idx[:, :cutoff_idx]
            day_edge_attr = day_edge_attr[:cutoff_idx, :]

            # Pass 1 day graph through model, save spatial embeddings
            spatial_embeddings = []

            # First GNN Layer
            x = self.gat1(day_node_feat, day_edge_idx, day_edge_attr)  # x becomes [10, self.emb_dim]
            spatial_embeddings.append(x)  # Append before ReLU(), straight from Graphgat layer
            # x = self.batch_norm_1(x)
            x = F.relu(x)
            
            x = self.gat2(x, day_edge_idx, day_edge_attr)  # x becomes [10, self.emb_dim]
            spatial_embeddings.append(x)
            # x = self.batch_norm_2(x)
            x = F.relu(x)

            # Concatenate spatial feature outputs of both graphgat layers; local and more global spatial info
            x = torch.cat(spatial_embeddings, dim=1)  # x becomes [10, 2 * self.emb_dim]

            # Apply TSNE reductionality to graphgat 1, graphgat 2, and concatenated spatial embeddings
            gat1_reduced_emb = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(spatial_embeddings[0].detach().numpy())  # [10, 2]
            gat2_reduced_emb = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(spatial_embeddings[1].detach().numpy())
            gat_concat_reduced_emb = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(x.detach().numpy())
            gat1_emb_list.append(np.expand_dims(gat1_reduced_emb, axis=0))
            gat2_emb_list.append(np.expand_dims(gat2_reduced_emb, axis=0))
            gat_concat_emb_list.append(np.expand_dims(gat_concat_reduced_emb, axis=0))
        
        # Concatenate embedding lists into single numpy arrays
        gat1_emb_list = np.concatenate(gat1_emb_list, axis=0)
        gat2_emb_list = np.concatenate(gat2_emb_list, axis=0)
        gat_concat_emb_list = np.concatenate(gat_concat_emb_list, axis=0)

        # Plot embeddings
        if not os.path.exists(os.path.join(save_path, "spatial_emb_plots")):
            os.mkdir(os.path.join(save_path, "spatial_emb_plots"))
        
        plot_tsne_reduced_embeddings(gat1_emb_list, plot_title="GAT Layer 1 Embeddings Epoch " + str(epoch_idx) + " " + split_name + " Dataset", save_filename=split_name + "_gat1_emb_epoch_" + str(epoch_idx), SAVE_PATH=os.path.join(save_path, "spatial_emb_plots"))
        plot_tsne_reduced_embeddings(gat2_emb_list, plot_title="GAT Layer 2 Embeddings Epoch " + str(epoch_idx) + " " + split_name + " Dataset", save_filename=split_name + "_gat2_emb_epoch_" + str(epoch_idx), SAVE_PATH=os.path.join(save_path, "spatial_emb_plots"))
        plot_tsne_reduced_embeddings(gat_concat_emb_list, plot_title="GAT Concat Embeddings Epoch " + str(epoch_idx) + " " + split_name + " Dataset", save_filename=split_name + "_concat_emb_epoch_" + str(epoch_idx), SAVE_PATH=os.path.join(save_path, "spatial_emb_plots"))


    def visualize_activations(self, args, data_loader, split_name, save_path, epoch_idx, color="C0"):
        gs_1_activations_list = [[] for _ in range(args["window"])]
        graphnorm_1_activations_list = [[] for _ in range(args["window"])]
        gs_2_activations_list = [[] for _ in range(args["window"])]
        graphnorm_2_activations_list = [[] for _ in range(args["window"])]
        gs_concat_activations_list = [[] for _ in range(args["window"])]
        lstm_h1_activations_list = [[] for _ in range(args["window"])]
        lstm_c1_activations_list = [[] for _ in range(args["window"])]
        lstm_h2_activations_list = [[] for _ in range(args["window"])]
        lstm_c2_activations_list = [[] for _ in range(args["window"])]
        concat_skip_activations = []
        mlp_lin1_activations = []
        mlp_lin2_activations = []

        self.eval()
        with torch.no_grad():
            for batch_window_node_feat, batch_window_edge_idx, batch_window_edge_attr, batch_window_labels in data_loader:
                for count, window_idx in enumerate(range(len(batch_window_node_feat))):
                    window_node_feat = batch_window_node_feat[0]  # shape [14, 10, 3]
                    window_edge_idx = batch_window_edge_idx[0]  # shape [14, 2, 100]
                    window_edge_attr = batch_window_edge_attr[0]  # shape [14, 100, 1]

                    h_1, c_1, h_2, c_2 = None, None, None, None

                    for day_idx in range(len(window_node_feat)):
                        day_node_feat = window_node_feat[day_idx]  # shape [10, 3]
                        day_edge_idx = window_edge_idx[day_idx]  # shape [2, 100]
                        day_edge_attr = window_edge_attr[day_idx]  # shape [100, 1]

                        # Remove -1s from edge connections and edge attributes => dynamic adjacency, connections specific for this day
                        cutoff_idx = day_edge_idx[0].tolist().index(-1)
                        day_edge_idx = day_edge_idx[:, :cutoff_idx]
                        day_edge_attr = day_edge_attr[:cutoff_idx, :]
                        
                        graph_outputs = []
                        if day_idx == 0:
                            self.concat_feat_list.append(day_node_feat)
                        else:
                            self.concat_feat_list.append(day_node_feat[:,2:3])

                        # First GNN Layer
                        x = self.gat1(day_node_feat, day_edge_idx, day_edge_attr)
                        gs_1_activations_list[day_idx].append(x.view(-1).numpy())
                        
                        x = self.graph_norm_1(x)
                        graphnorm_1_activations_list[day_idx].append(x.view(-1).numpy())

                        x = F.relu(x)
                        graph_outputs.append(x)
                        
                        x = self.gat2(x, day_edge_idx, day_edge_attr)
                        gs_2_activations_list[day_idx].append(x.view(-1).numpy())
                        
                        x = self.graph_norm_2(x)
                        graphnorm_2_activations_list[day_idx].append(x.view(-1).numpy())

                        x = F.relu(x)
                        graph_outputs.append(x)

                        x = torch.cat(graph_outputs, dim=1)
                        gs_concat_activations_list[day_idx].append(x.view(-1).numpy())

                        if h_1 is None:
                            h_1 = torch.zeros(x.shape[0], self.emb_dim)
                        if c_1 is None:
                            c_1 = torch.zeros(x.shape[0], self.emb_dim)
                        if h_2 is None:
                            h_2 = torch.zeros(x.shape[0], self.emb_dim)
                        if c_2 is None:
                            c_2 = torch.zeros(x.shape[0], self.emb_dim)

                        # RNN Layer
                        h_1, c_1 = self.lstm1(x, (h_1, c_1))
                        lstm_h1_activations_list[day_idx].append(h_1.view(-1).numpy())
                        lstm_c1_activations_list[day_idx].append(c_1.view(-1).numpy())
                        
                        h_2, c_2 = self.lstm2(h_1, (h_2, c_2))
                        lstm_h2_activations_list[day_idx].append(h_2.view(-1).numpy())
                        lstm_c2_activations_list[day_idx].append(c_2.view(-1).numpy())

                        if day_idx == self.window_size - 1:
                            # Skip connection
                            concat_feat = torch.cat(self.concat_feat_list, dim=1)
                            x = torch.cat((concat_feat, h_1, h_2), dim=1)
                            concat_skip_activations.append(x.view(-1).numpy())
                            self.concat_feat_list.clear()

                            # Readout and activation layers
                            x = self.act1(x)
                            x = self.lin1(x)
                            mlp_lin1_activations.append(x.view(-1).numpy())
                            # x = self.act2(x)
                            # x = self.lin2(x)
                            # mlp_lin2_activations.append(x.view(-1).numpy())

        # Plot activation distributions
        activation_distrib_plots = os.path.join(save_path, "activation_distrib_plots")
        if not os.path.exists(activation_distrib_plots):
            os.mkdir(activation_distrib_plots)
        
        activations = {}
        activations["GAT1"] = np.array(gs_1_activations_list).flatten()
        activations["GraphNorm1"] = np.array(graphnorm_1_activations_list).flatten()
        activations["GAT2"] = np.array(gs_2_activations_list).flatten()
        activations["GraphNorm2"] = np.array(graphnorm_2_activations_list).flatten()
        activations["GATConcat"] = np.array(gs_concat_activations_list).flatten()
        activations["LSTM_h1"] = np.array(lstm_h1_activations_list).flatten()
        activations["LSTM_c1"] = np.array(lstm_c1_activations_list).flatten()
        activations["LSTM_h2"] = np.array(lstm_h2_activations_list).flatten()
        activations["LSTM_c2"] = np.array(lstm_c2_activations_list).flatten()
        activations["Concat"] = np.array(concat_skip_activations).flatten()
        activations["MLP_lin1"] = np.array(mlp_lin1_activations).flatten()
        activations["MLP_lin2"] = np.array(mlp_lin2_activations).flatten()

        columns = 4
        rows = math.ceil(len(activations)/columns)
        fig, ax = plt.subplots(rows, columns, figsize=(columns*2.7, rows*2.5))
        fig_index = 0
        for key in activations:
            key_ax = ax[fig_index//columns][fig_index%columns]
            sns.histplot(data=activations[key], bins=50, ax=key_ax, color=color, kde=True, stat="density")
            key_ax.set_title(f"Layer {key}")
            fig_index += 1
        fig.suptitle(f"Activation distribution on 14th Day of First Window of " + split_name + " Dataset ", fontsize=14)
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        # plt.show()
        plt.savefig(os.path.join(activation_distrib_plots, split_name + "_activations_epoch_" + str(epoch_idx)))
        plt.clf()
        plt.close()

