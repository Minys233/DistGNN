import torch
import torch_geometric.nn as pygnn
from torch import nn
from torch_scatter import scatter_add


# for model develop
class GlobalAddPool(nn.Module):
    def forward(self, x, batch, *args):
        return pygnn.global_add_pool(x, batch)


# model develop
class NodeAttentionPool(nn.Module):
    def __init__(self, gate_nn, nn):
        super().__init__()
        self.gate_nn = gate_nn
        self.nn = nn

    def forward(self, x, batch, *args, return_score=False):
        size = batch[-1].item() + 1
        gate = self.gate_nn(x).view(-1, 1)
        x = self.nn(x)
        gate = torch.sigmoid(gate)
        out = scatter_add(gate * x, batch, dim=0, dim_size=size)
        if return_score:
            return out, gate
        else:
            return out

    def __repr__(self):
        return '{}(gate_nn={}, nn={})'.format(self.__class__.__name__, self.gate_nn, self.nn)


# for final model
class StructureAttentionPool(nn.Module):
    def __init__(self, num_features):
        super(StructureAttentionPool, self).__init__()
        self.num_features = num_features
        self.fc = nn.Linear(num_features, num_features)

    def forward(self, x, batch, *args, return_score=False):
        ctx = self.fc(pygnn.global_mean_pool(x, batch))
        ctx = torch.tanh(ctx)  # (numG, num_features)
        repeats = scatter_add(torch.ones(batch.size()[0], dtype=torch.long).to(batch.device), batch)
        ctx = torch.repeat_interleave(ctx, repeats, dim=0)  # (numN, num_features)
        score = torch.bmm(x.view(x.size()[0], 1, x.size()[1]), ctx.view(ctx.size()[0], ctx.size()[1], 1)).squeeze()
        score = torch.sigmoid(score).view(-1, 1)  # (numN, 1)
        out = pygnn.global_add_pool(score * x, batch)
        if return_score:
            return out, score
        else:
            return out

    def __repr__(self):
        return '{}(num_features={}, ctx_nn={})'.format(self.__class__.__name__,
                                                       self.num_features, self.fc)
