import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn.norm import GraphNorm
from models.weight_sage import WeightedSAGEConv

import os
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.manifold import TSNE
from utils.training_utils import plot_tsne_reduced_embeddings

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DynamicAdjSAGE(torch.nn.Module):
    """
    Arguments:
        node_features: number of features each node in 1-day graph contains
        emd_dim: embedding dimension that WeightedGraphSAGE layers will output
        window_size: number of 1-day graphs that will be passed through network before predicting next day
        output: number of features to predict for each node on N+1 day
        training: whether the model is being loaded for training or test. Affects things like dropout if present.
        lstm_type: what type of LSTM to use (["vanila"])
    """

    def __init__(self,
                node_features: int = 2, 
                emb_dim: int = 10,
                window_size: int = 14,
                output: int = 1, 
                training: bool = True,
                lstm_type: str = 'vanilla',
                name: str = "DASAGE"):
        super(DynamicAdjSAGE, self).__init__()
        assert lstm_type in ["vanilla"]

        self.emb_dim = emb_dim
        self.window_size = window_size
        self.training = training
        self.lstm_type = lstm_type
        self.name = name

        normalize_graphsage_layers = False

        self.sage1 = WeightedSAGEConv(in_channels=node_features, out_channels=self.emb_dim, normalize=normalize_graphsage_layers, training=self.training)
        self.sage2 = WeightedSAGEConv(in_channels=self.emb_dim, out_channels=self.emb_dim, normalize=normalize_graphsage_layers, training=self.training)
        
        self.graph_norm_1 = GraphNorm(self.emb_dim)
        self.graph_norm_2 = GraphNorm(self.emb_dim)

        self.lstm1 = nn.LSTMCell(input_size=2 * self.emb_dim, hidden_size=self.emb_dim)
        self.lstm2 = nn.LSTMCell(input_size=self.emb_dim, hidden_size=self.emb_dim)
        
        self.act1 = torch.nn.ReLU()
        self.lin1 = torch.nn.Linear(self.window_size + (2 * self.emb_dim), output)

        # self.init_weights()  # Initialize weights with orthogonal matrices
        self.concat_feat_list = []
    
    def init_weights(self):
        nn.init.orthogonal_(self.sage1.lin_l.weight)
        nn.init.orthogonal_(self.sage1.lin_r.weight)
        nn.init.orthogonal_(self.sage2.lin_l.weight)
        nn.init.orthogonal_(self.sage2.lin_r.weight)
        # nn.init.orthogonal_(self.graph_norm_1.weight)  # Only tensors with 2+ dimensions are supported
        # nn.init.orthogonal_(self.graph_norm_2.weight)

        # Pytorch LSTMCell only has 2 weight matrices, each one is 4*hidden_size * output_size,
        # meaning these 2 matrices contain the 8 LSTM kernels we are trying to initialize
        nn.init.orthogonal_(self.lstm1.weight_ih)
        nn.init.orthogonal_(self.lstm1.weight_hh)
        nn.init.orthogonal_(self.lstm2.weight_ih)
        nn.init.orthogonal_(self.lstm2.weight_hh)

        nn.init.orthogonal_(self.lin1.weight)
        nn.init.orthogonal_(self.lin2.weight)
        print("Ran init_weights().")

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
        
        graphsage_outputs = []
        self.concat_feat_list.append(x)  # [:,1:2]

        # First GNN Layer
        x = self.sage1(x, edge_index=edge_index, edge_weight=edge_attr)
        x = self.graph_norm_1(x)
        x = F.relu(x)
        graphsage_outputs.append(x)
        
        x = self.sage2(x, edge_index=edge_index, edge_weight=edge_attr)
        x = self.graph_norm_2(x)
        x = F.relu(x)
        graphsage_outputs.append(x)

        x = torch.cat(graphsage_outputs, dim=1)

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

        return x, h_1, c_1, h_2, c_2


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
        sage1_emb_list = []
        sage2_emb_list = []
        sage_concat_emb_list = []
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
            x = self.sage1(day_node_feat, day_edge_idx, day_edge_attr)  # x becomes [10, self.emb_dim]
            spatial_embeddings.append(x)  # Append before ReLU(), straight from GraphSAGE layer
            # x = self.batch_norm_1(x)
            x = F.relu(x)
            
            x = self.sage2(x, day_edge_idx, day_edge_attr)  # x becomes [10, self.emb_dim]
            spatial_embeddings.append(x)
            # x = self.batch_norm_2(x)
            x = F.relu(x)

            # Concatenate spatial feature outputs of both graphsage layers; local and more global spatial info
            x = torch.cat(spatial_embeddings, dim=1)  # x becomes [10, 2 * self.emb_dim]

            # Apply TSNE reductionality to graphsage 1, graphsage 2, and concatenated spatial embeddings
            graphsage1_reduced_emb = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(spatial_embeddings[0].detach().numpy())  # [10, 2]
            graphsage2_reduced_emb = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(spatial_embeddings[1].detach().numpy())
            graphsage_concat_reduced_emb = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(x.detach().numpy())
            sage1_emb_list.append(np.expand_dims(graphsage1_reduced_emb, axis=0))
            sage2_emb_list.append(np.expand_dims(graphsage2_reduced_emb, axis=0))
            sage_concat_emb_list.append(np.expand_dims(graphsage_concat_reduced_emb, axis=0))
        
        # Concatenate embedding lists into single numpy arrays
        sage1_emb_list = np.concatenate(sage1_emb_list, axis=0)
        sage2_emb_list = np.concatenate(sage2_emb_list, axis=0)
        sage_concat_emb_list = np.concatenate(sage_concat_emb_list, axis=0)

        # Plot embeddings
        if not os.path.exists(os.path.join(save_path, "spatial_emb_plots")):
            os.mkdir(os.path.join(save_path, "spatial_emb_plots"))
        
        plot_tsne_reduced_embeddings(sage1_emb_list, plot_title="GraphSAGE Layer 1 Embeddings Epoch " + str(epoch_idx) + " " + split_name + " Dataset", save_filename=split_name + "_sage1_emb_epoch_" + str(epoch_idx), SAVE_PATH=os.path.join(save_path, "spatial_emb_plots"))
        plot_tsne_reduced_embeddings(sage2_emb_list, plot_title="GraphSAGE Layer 2 Embeddings Epoch " + str(epoch_idx) + " " + split_name + " Dataset", save_filename=split_name + "_sage2_emb_epoch_" + str(epoch_idx), SAVE_PATH=os.path.join(save_path, "spatial_emb_plots"))
        plot_tsne_reduced_embeddings(sage_concat_emb_list, plot_title="GraphSAGE Concat Embeddings Epoch " + str(epoch_idx) + " " + split_name + " Dataset", save_filename=split_name + "_concat_emb_epoch_" + str(epoch_idx), SAVE_PATH=os.path.join(save_path, "spatial_emb_plots"))


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
                        
                        graphsage_outputs = []
                        if day_idx == 0:
                            self.concat_feat_list.append(day_node_feat)
                        else:
                            self.concat_feat_list.append(day_node_feat[:,2:3])

                        # First GNN Layer
                        x = self.sage1(day_node_feat, day_edge_idx, day_edge_attr)
                        gs_1_activations_list[day_idx].append(x.view(-1).numpy())
                        
                        x = self.graph_norm_1(x)
                        graphnorm_1_activations_list[day_idx].append(x.view(-1).numpy())

                        x = F.relu(x)
                        graphsage_outputs.append(x)
                        
                        x = self.sage2(x, day_edge_idx, day_edge_attr)
                        gs_2_activations_list[day_idx].append(x.view(-1).numpy())
                        
                        x = self.graph_norm_2(x)
                        graphnorm_2_activations_list[day_idx].append(x.view(-1).numpy())

                        x = F.relu(x)
                        graphsage_outputs.append(x)

                        x = torch.cat(graphsage_outputs, dim=1)
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
        activations["Sage1"] = np.array(gs_1_activations_list).flatten()
        activations["GraphNorm1"] = np.array(graphnorm_1_activations_list).flatten()
        activations["Sage2"] = np.array(gs_2_activations_list).flatten()
        activations["GraphNorm2"] = np.array(graphnorm_2_activations_list).flatten()
        activations["SageConcat"] = np.array(gs_concat_activations_list).flatten()
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

