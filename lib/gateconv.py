from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (Adj, Size, OptTensor)
from torch_geometric.utils import softmax


# adapted from pyg.nn.GATConv, add edge channel in attention computing
# reference: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gat_conv.html
class GATEConv(MessagePassing):
    _alpha: OptTensor

    def __init__(self, in_channels: Tuple[int, int], out_channels: Tuple[int, int],
                 heads: int = 1, concat: bool = True, negative_slope: float = 0.2,
                 dropout: float = 0.0, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GATEConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin_l = Linear(in_channels[0], heads * out_channels[0], bias=False)
        self.lin_r = self.lin_l
        self.lin_e = Linear(in_channels[1], heads * out_channels[1], bias=False)

        self.att_l = Parameter(torch.Tensor(1, heads, out_channels[0]))
        self.att_r = Parameter(torch.Tensor(1, heads, out_channels[0]))
        self.att_e = Parameter(torch.Tensor(1, heads, out_channels[1]))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels[0]))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels[0]))
        else:
            self.register_parameter('bias', None)

        self._alpha = None
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        glorot(self.lin_e.weight)
        glorot(self.att_l)
        glorot(self.att_r)
        glorot(self.att_e)
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor, edge_embedding,
                size: Size = None, return_attention_weights=None):
        H, C = self.heads, self.out_channels[0]
        assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
        x_l = x_r = self.lin_l(x).view(-1, H, C)
        alpha_l = (x_l * self.att_l).sum(dim=-1)
        alpha_r = (x_r * self.att_r).sum(dim=-1)
        ee = self.lin_e(edge_embedding).view(-1, self.heads, self.out_channels[1])
        alpha_e = (ee * self.att_e).sum(dim=-1)

        assert x_l is not None
        assert alpha_l is not None
        # assume graphs have self loop
        out = self.propagate(edge_index, alpha_e=alpha_e, edge_attr=edge_attr,
                             x=(x_l, x_r), alpha=(alpha_l, alpha_r), size=size)
        alpha = self._alpha
        self._alpha = None
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels[0])
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            return out, (edge_index, alpha)
        else:
            return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, alpha_e, edge_attr: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        alpha = (alpha_j + alpha_i) + alpha_e[edge_attr]
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
