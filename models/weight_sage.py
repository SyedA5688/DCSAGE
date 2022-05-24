import torch.nn.functional as F

from typing import Union, Tuple
from torch import Tensor, matmul
from torch.nn import Linear
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import OptPairTensor, Adj, Size


class WeightedSAGEConv(MessagePassing):
    """The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper

    Copied from torch_geometric.nn.SageConv and then modified by Sesti et. al and Juan
    Jose Garau to take edge weights into account in message-passing step.

    math:
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W_2} \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j

    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to True, output features
            will be l_2-normalized (default: False).
        bias (bool, optional): If set to False, the layer will not learn
            an additive bias. (default: True)
        **kwargs (optional): Additional arguments of
            torch_geometric.nn.conv.MessagePassing.
    """

    def __init__(self, 
                in_channels: Union[int, Tuple[int, int]],
                out_channels: int, 
                normalize: bool = False,
                training: bool = True,
                root_weight = True,
                bias: bool = True, 
                **kwargs):
        super(WeightedSAGEConv, self).__init__(aggr='mean', **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.training = training

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)
        
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_weight: Tensor = None,
                size: Size = None) -> Tensor:

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size, edge_weight=edge_weight)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_weight) -> Tensor:
        """
        Constructs messages from node j to node i in analogy to ϕΘ for each edge in 
        edge_index. This function can take any argument as input which was initially 
        passed to propagate(). Furthermore, tensors passed to propagate() can be 
        mapped to the respective nodes i and j by appending _i or _j to the variable 
        name, .e.g. x_i and x_j.

        x_i.shape and x_j.shape is [num_edges, embedding dim (num_features or graph emb dim)]
        edge_weight.shape is [num_edges, 1]
        """

        return x_j * edge_weight  # [num_edges, dim] * [num_edges, 1] = [num_edges, dim]
        # return x_j

    # def message_and_aggregate(self, adj_t: SparseTensor, x: OptPairTensor) -> Tensor:
    #     # Not using Sparse Tensors, so this is not called
    #     adj_t = adj_t.set_value(None, layout=None)
    #     return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)
