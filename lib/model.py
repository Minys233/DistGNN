import torch
from torch import nn
from torch.nn.utils import rnn
import torch.nn.functional as F
import torch_geometric as pyg
import torch_geometric.nn as pygnn

from lib.gateconv import GATEConv
from lib.dataset import ProteinData


class GATNet(nn.Module):
    def __init__(self, enum, hid, fout, convlayer, drop, actfn, gat_heads, edge_emb, jk_layer, **kwargs):
        # useless parameters are for compatibility
        super(GATNet, self).__init__()
        self.emb_l = nn.ModuleList()
        self.conv_l = nn.ModuleList()
        self.jk_l = None
        self.dummy_rnn_l = None

        self.emb_r = nn.ModuleList()
        self.conv_r = nn.ModuleList()
        self.jk_r = None
        self.dummy_rnn_r = None

        self.finals = nn.ModuleList()

        self.hid = hid
        self.fout = fout
        self.convlayer = convlayer
        self.drop = drop
        self.actfn = actfn
        self.gat_heads = gat_heads
        self.jk_layer = jk_layer
        self.setup(**kwargs)

    def setup(self, **kwargs):
        self.emb_l.append(nn.Embedding(len(ProteinData.AA_3LETTER), self.hid - 5))
        self.emb_l.append(nn.Embedding(len(ProteinData.DSSP_SS3), self.hid - 5))

        self.emb_r.append(nn.Embedding(len(ProteinData.AA_3LETTER), self.hid - 5))
        self.emb_r.append(nn.Embedding(len(ProteinData.DSSP_SS3), self.hid - 5))

        for idx in range(self.convlayer):
            self.conv_l.append(pygnn.GATConv(self.hid, self.hid, heads=self.gat_heads, dropout=self.drop, concat=False))
            self.conv_r.append(pygnn.GATConv(self.hid, self.hid, heads=self.gat_heads, dropout=self.drop, concat=False))

        self.jk_l = pygnn.JumpingKnowledge("lstm", self.hid, num_layers=self.jk_layer)
        self.jk_r = pygnn.JumpingKnowledge("lstm", self.hid, num_layers=self.jk_layer)
        self.dummy_rnn_l = nn.Linear(self.hid, self.hid)
        self.dummy_rnn_r = nn.Linear(self.hid, self.hid)
        for i in range(4):
            self.finals.append(nn.Linear(self.hid, self.fout))
        self.finals.append(nn.Linear(self.fout, 1, bias=False))

    def encode(self, x, index, ptr):
        x_idx, x_flt = x
        index_l, index_r = index
        # embed
        x_l = torch.cat([self.emb_l[0](x_idx[:, 0]) + self.emb_l[1](x_idx[:, 1]), x_flt], dim=1)
        x_r = torch.cat([self.emb_r[0](x_idx[:, 0]) + self.emb_r[1](x_idx[:, 1]), x_flt], dim=1)
        # drop edge
        if self.training:
            index_l, _ = pyg.utils.dropout_adj(index_l, p=self.drop, force_undirected=True)
            index_r, _ = pyg.utils.dropout_adj(index_r, p=self.drop, force_undirected=True)
        # drop node
        x_l = F.dropout(x_l, p=self.drop, training=self.training)
        x_r = F.dropout(x_r, p=self.drop, training=self.training)
        # pass network
        xs_l, xs_r = [], []
        for idx in range(self.convlayer):
            o_l = (x_l + x_r) / 2 + self.actfn(self.conv_l[idx](x_l, index_l))
            o_r = (x_r + x_l) / 2 + self.actfn(self.conv_r[idx](x_r, index_r))
            x_l = F.dropout(o_l, p=self.drop, training=self.training)
            x_r = F.dropout(o_r, p=self.drop, training=self.training)
            xs_l.append(x_l)
            xs_r.append(x_r)

        jk_l, jk_r = self.jk_l(xs_l), self.jk_r(xs_r)  # topology embedding
        sq_l, sq_r = self.dummy_rnn_l(jk_l), self.dummy_rnn_r(jk_r)
        return jk_l, sq_l, jk_r, sq_r

    def forward(self, batch):
        x_idx, x_flt, index_l, index_r = batch.x, batch.x_flt, batch.edge_index_l, batch.edge_index_r
        to_l, sq_l, to_r, sq_r = self.encode((x_idx, x_flt), (index_l, index_r), batch.ptr)
        to_l, sq_l, to_r, sq_r = [self.finals[i](xx) for i, xx in enumerate([to_l, sq_l, to_r, sq_r])]
        # here sequential embedding is dummy
        alpha = torch.stack([self.finals[-1](arr) for arr in [to_l, sq_l, to_r, sq_r]], dim=1)
        alpha = F.softmax(alpha, dim=1)  # Nnode, 4
        nodeemb = to_l * alpha[:, 0].view(-1, 1) + sq_l * alpha[:, 1].view(-1, 1) + \
                  to_r * alpha[:, 2].view(-1, 1) + sq_r * alpha[:, 3].view(-1, 1)
        return nodeemb


class GATENet(nn.Module):
    def __init__(self, enum, hid, fout, convlayer, drop, actfn, gat_heads, edge_emb, jk_layer, **kwargs):
        super(GATENet, self).__init__()
        self.emb_l = nn.ModuleList()
        self.conv_l = nn.ModuleList()
        self.jk_l = None
        self.dummy_rnn_l = None
        self.edgenet_l = nn.ModuleList()

        self.emb_r = nn.ModuleList()
        self.conv_r = nn.ModuleList()
        self.jk_r = None
        self.dummy_rnn_r = None
        self.edgenet_r = nn.ModuleList()

        self.finals = nn.ModuleList()
        # self.edge_emb_tmp = None

        self.enum = enum
        self.hid = hid
        self.fout = fout
        self.convlayer = convlayer
        self.drop = drop
        self.actfn = actfn
        self.gat_heads = gat_heads
        self.edge_emb = edge_emb
        self.jk_layer = jk_layer
        self.setup(**kwargs)

    def setup(self, **kwargs):
        self.emb_l.append(nn.Embedding(len(ProteinData.AA_3LETTER), self.hid - 5))
        self.emb_l.append(nn.Embedding(len(ProteinData.DSSP_SS3), self.hid - 5))
        self.emb_l.append(nn.Embedding(self.enum, self.edge_emb))

        self.emb_r.append(nn.Embedding(len(ProteinData.AA_3LETTER), self.hid - 5))
        self.emb_r.append(nn.Embedding(len(ProteinData.DSSP_SS3), self.hid - 5))
        self.emb_r.append(nn.Embedding(self.enum, self.edge_emb))

        for idx in range(self.convlayer):
            self.conv_l.append(
                GATEConv((self.hid, self.edge_emb), (self.hid, self.edge_emb), dropout=self.drop,
                         heads=self.gat_heads, concat=False))
            self.conv_r.append(
                GATEConv((self.hid, self.edge_emb), (self.hid, self.edge_emb), dropout=self.drop,
                         heads=self.gat_heads, concat=False))
            self.edgenet_l.append(nn.Linear(self.edge_emb, self.edge_emb))
            self.edgenet_r.append(nn.Linear(self.edge_emb, self.edge_emb))
        self.jk_l = pygnn.JumpingKnowledge("lstm", self.hid, num_layers=self.jk_layer)
        self.jk_r = pygnn.JumpingKnowledge("lstm", self.hid, num_layers=self.jk_layer)
        self.dummy_rnn_l = nn.Linear(self.hid, self.hid)
        self.dummy_rnn_r = nn.Linear(self.hid, self.hid)
        for i in range(4):
            self.finals.append(nn.Linear(self.hid, self.fout))
        self.finals.append(nn.Linear(self.fout, 1, bias=False))

    def encode(self, x, index, attr, ptr):
        x_idx, x_flt = x
        index_l, index_r = index
        attr_l, attr_r = attr
        # embed
        x_l = torch.cat([self.emb_l[0](x_idx[:, 0]) + self.emb_l[1](x_idx[:, 1]), x_flt], dim=1)
        x_r = torch.cat([self.emb_r[0](x_idx[:, 0]) + self.emb_r[1](x_idx[:, 1]), x_flt], dim=1)
        # drop edge
        if self.training:
            index_l, attr_l = pyg.utils.dropout_adj(index_l, attr_l, p=self.drop, force_undirected=True)
            index_r, attr_r = pyg.utils.dropout_adj(index_r, attr_r, p=self.drop, force_undirected=True)
        tmp = torch.arange(self.enum, dtype=torch.long, device=self.emb_l[2].weight.device)
        edge_embedding_l = self.emb_l[2](tmp)
        edge_embedding_r = self.emb_r[2](tmp)
        # drop node
        x_l = F.dropout(x_l, p=self.drop, training=self.training)
        x_r = F.dropout(x_r, p=self.drop, training=self.training)
        # pass network
        xs_l, xs_r = [], []
        for idx in range(self.convlayer):
            o_l = (x_l + x_r) / 2 + self.actfn(self.conv_l[idx](x_l, index_l, attr_l, edge_embedding_l))
            o_r = (x_r + x_l) / 2 + self.actfn(self.conv_r[idx](x_r, index_r, attr_r, edge_embedding_r))
            x_l = F.dropout(o_l, p=self.drop, training=self.training)
            x_r = F.dropout(o_r, p=self.drop, training=self.training)
            edge_embedding_l = self.edgenet_l[idx](edge_embedding_l)
            edge_embedding_r = self.edgenet_r[idx](edge_embedding_r)
            xs_l.append(x_l)
            xs_r.append(x_r)

        jk_l, jk_r = self.jk_l(xs_l), self.jk_r(xs_r)  # topology embedding
        sq_l, sq_r = self.dummy_rnn_l(jk_l), self.dummy_rnn_r(jk_r)
        return jk_l, sq_l, jk_r, sq_r

    def forward(self, batch):
        x_idx, x_flt, index_l, index_r = batch.x, batch.x_flt, batch.edge_index_l, batch.edge_index_r
        attr_l, attr_r = batch.edge_attr_l, batch.edge_attr_r
        to_l, sq_l, to_r, sq_r = self.encode((x_idx, x_flt), (index_l, index_r), (attr_l, attr_r), batch.ptr)
        to_l, sq_l, to_r, sq_r = [self.finals[i](xx) for i, xx in enumerate([to_l, sq_l, to_r, sq_r])]
        # here sequential embedding is dummy
        alpha = torch.stack([self.finals[-1](arr) for arr in [to_l, sq_l, to_r, sq_r]], dim=1)
        alpha = F.softmax(alpha, dim=1)  # Nnode, 4
        nodeemb = to_l * alpha[:, 0].view(-1, 1) + sq_l * alpha[:, 1].view(-1, 1) + \
                  to_r * alpha[:, 2].view(-1, 1) + sq_r * alpha[:, 3].view(-1, 1)
        return nodeemb


class LSTMNet(nn.Module):
    def __init__(self, enum, hid, fout, convlayer, drop, actfn, gat_heads, edge_emb, jk_layer, **kwargs):
        super(LSTMNet, self).__init__()
        self.emb_l = nn.ModuleList()
        self.conv_l = nn.ModuleList()
        self.rnn_l = nn.ModuleList()
        self.edgenet_l = nn.ModuleList()
        self.jk_l = None

        self.emb_r = nn.ModuleList()
        self.conv_r = nn.ModuleList()
        self.rnn_r = nn.ModuleList()
        self.edgenet_r = nn.ModuleList()
        self.jk_r = None

        self.finals = nn.ModuleList()

        self.enum = enum
        self.hid = hid
        self.fout = fout
        self.convlayer = convlayer
        self.drop = drop
        self.actfn = actfn
        self.gat_heads = gat_heads
        self.edge_emb = edge_emb
        self.jk_layer = jk_layer
        self.setup(**kwargs)

    def setup(self, **kwargs):
        kwargs.setdefault('lstm_layer', 1)
        self.emb_l.append(nn.Embedding(len(ProteinData.AA_3LETTER), self.hid - 5))
        self.emb_l.append(nn.Embedding(len(ProteinData.DSSP_SS3), self.hid - 5))
        self.emb_l.append(nn.Embedding(self.enum, self.edge_emb))

        self.emb_r.append(nn.Embedding(len(ProteinData.AA_3LETTER), self.hid - 5))
        self.emb_r.append(nn.Embedding(len(ProteinData.DSSP_SS3), self.hid - 5))
        self.emb_r.append(nn.Embedding(self.enum, self.edge_emb))

        for idx in range(self.convlayer):
            self.conv_l.append(
                GATEConv((self.hid, self.edge_emb), (self.hid, self.edge_emb), dropout=self.drop,
                         heads=self.gat_heads, concat=False))
            self.conv_r.append(
                GATEConv((self.hid, self.edge_emb), (self.hid, self.edge_emb), dropout=self.drop,
                         heads=self.gat_heads, concat=False))
            self.edgenet_l.append(nn.Linear(self.edge_emb, self.edge_emb))
            self.edgenet_r.append(nn.Linear(self.edge_emb, self.edge_emb))
        self.rnn_l.append(
            nn.LSTM(self.hid, self.hid // 2, num_layers=kwargs['lstm_layer'], batch_first=True, bidirectional=True))
        self.rnn_r.append(
            nn.LSTM(self.hid, self.hid // 2, num_layers=kwargs['lstm_layer'], batch_first=True, bidirectional=True))
        self.jk_l = pygnn.JumpingKnowledge("lstm", self.hid, num_layers=self.jk_layer)
        self.jk_r = pygnn.JumpingKnowledge("lstm", self.hid, num_layers=self.jk_layer)

        for i in range(4):
            self.finals.append(nn.Linear(self.hid, self.fout))
        self.finals.append(nn.Linear(self.fout, 1, bias=False))

    def lstm_pass(self, lstm, x, ptr):
        seqs = []
        for s, e in zip(ptr[:-1], ptr[1:]):
            seqs.append(x[s:e])
        packed = rnn.pack_sequence(seqs, enforce_sorted=False)
        out, (h_n, c_n) = lstm(packed)
        # h_n: layer*2, Nseq, hidden
        # h_n = h_n.permute(1, 0, 2)
        padded, lens = rnn.pad_packed_sequence(out, batch_first=True)
        nodes = torch.cat([padded[i, :lens[i]] for i in range(len(lens))])
        return nodes

    def forward(self, batch):
        x_idx, x_flt, index_l, index_r = batch.x, batch.x_flt, batch.edge_index_l, batch.edge_index_r
        attr_l, attr_r = batch.edge_attr_l, batch.edge_attr_r
        to_l, sq_l, to_r, sq_r = self.encode((x_idx, x_flt), (index_l, index_r), (attr_l, attr_r), batch.ptr)
        to_l, sq_l, to_r, sq_r = [self.finals[i](xx) for i, xx in enumerate([to_l, sq_l, to_r, sq_r])]
        alpha = torch.stack([self.finals[-1](arr) for arr in [to_l, sq_l, to_r, sq_r]], dim=1)
        alpha = F.softmax(alpha, dim=1)  # Nnode, 4
        nodeemb = to_l * alpha[:, 0].view(-1, 1) + sq_l * alpha[:, 1].view(-1, 1) + \
                  to_r * alpha[:, 2].view(-1, 1) + sq_r * alpha[:, 3].view(-1, 1)
        return nodeemb

    def encode(self, x, index, attr, ptr):
        x_idx, x_flt = x
        index_l, index_r = index
        attr_l, attr_r = attr
        # embed
        x_l = torch.cat([self.emb_l[0](x_idx[:, 0]) + self.emb_l[1](x_idx[:, 1]), x_flt], dim=1)
        x_r = torch.cat([self.emb_r[0](x_idx[:, 0]) + self.emb_r[1](x_idx[:, 1]), x_flt], dim=1)
        tmp = torch.arange(self.enum, dtype=torch.long, device=self.emb_l[2].weight.device)
        edge_embedding_l = self.emb_l[2](tmp)
        edge_embedding_r = self.emb_r[2](tmp)
        if self.training:
            index_l, attr_l = pyg.utils.dropout_adj(index_l, attr_l, p=self.drop, force_undirected=True)
            index_r, attr_r = pyg.utils.dropout_adj(index_r, attr_r, p=self.drop, force_undirected=True)

        x_l = F.dropout(x_l, p=self.drop, training=self.training)
        x_r = F.dropout(x_r, p=self.drop, training=self.training)
        xs_l, xs_r = [], []

        for idx in range(self.convlayer):
            o_l = (x_l + x_r) / 2 + self.actfn(self.conv_l[idx](x_l, index_l, attr_l, edge_embedding_l))
            o_r = (x_r + x_l) / 2 + self.actfn(self.conv_r[idx](x_r, index_r, attr_r, edge_embedding_r))
            x_l = F.dropout(o_l, p=self.drop, training=self.training)
            x_r = F.dropout(o_r, p=self.drop, training=self.training)
            edge_embedding_l = self.edgenet_l[idx](edge_embedding_l)
            edge_embedding_r = self.edgenet_r[idx](edge_embedding_r)
            xs_l.append(x_l)
            xs_r.append(x_r)

        jk_l, jk_r = self.jk_l(xs_l), self.jk_r(xs_r)  # topology embedding
        x_l = self.lstm_pass(self.rnn_l[0], jk_l, ptr)
        x_r = self.lstm_pass(self.rnn_r[0], jk_r, ptr)  # sequential embedding
        return jk_l, x_l, jk_r, x_r



