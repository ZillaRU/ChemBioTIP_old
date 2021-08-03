import torch
import torch.nn as nn
import torch.nn.functional as F

# reference: GraIL
from model.inter_gnn import SumAggregator, MLPAggregator, GRUAggregator
from model.inter_gnn.grail_rgcn_layer import RGCNBasisLayer as RGCNLayer


class InterView_RGCN(nn.Module):
    def __init__(self,
                 inp_dim,
                 emb_dim,
                 attn_rel_emb_dim,
                 num_rels,
                 rel2id,
                 aug_num_rels,
                 num_bases,
                 num_gcn_layers,
                 dropout, edge_dropout,
                 gnn_agg_type,
                 has_attn,
                 device
                 ):
        super(InterView_RGCN, self).__init__()
        self.layers = nn.ModuleList()
        self.inp_dim = inp_dim
        self.emb_dim = emb_dim
        self.attn_rel_emb_dim = attn_rel_emb_dim
        self.num_rels = num_rels
        self.rel2id = rel2id
        self.aug_num_rels = aug_num_rels
        self.num_bases = num_bases
        self.num_hidden_layers = num_gcn_layers
        self.dropout = dropout
        self.edge_dropout = edge_dropout
        self.aggregator_type = gnn_agg_type
        self.has_attn = has_attn
        self.device = device

        if self.has_attn:
            self.attn_rel_emb = nn.Embedding(self.num_rels, self.attn_rel_emb_dim, sparse=False)
        else:
            self.attn_rel_emb = None

        # initialize aggregators for input and hidden layers
        if gnn_agg_type == "sum":
            self.aggregator = SumAggregator(self.emb_dim)
        elif gnn_agg_type == "mlp":
            self.aggregator = MLPAggregator(self.emb_dim)
        elif gnn_agg_type == "gru":
            self.aggregator = GRUAggregator(self.emb_dim)

        # initialize basis weights for input and hidden layers
        self.input_basis_weights = nn.Parameter(torch.Tensor(self.num_bases, self.inp_dim, self.emb_dim))
        self.basis_weights = nn.Parameter(torch.Tensor(self.num_bases, self.emb_dim, self.emb_dim))

        # create rgcn layers
        self.build_model()

        # create initial features
        self.features = self.create_features()

    def create_features(self):
        features = torch.arange(self.inp_dim).to(device=self.device)
        return features

    def build_model(self):
        # i2h
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
        # h2h
        for idx in range(self.num_hidden_layers - 1):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)

    def build_input_layer(self):
        return RGCNLayer(self.inp_dim,
                         self.emb_dim,
                         # self.input_basis_weights,
                         self.aggregator,
                         attn_rel_emb_dim=self.attn_rel_emb_dim,
                         num_rels=self.num_rels,
                         rel2id=self.rel2id,
                         device=self.device,
                         activation=F.relu,
                         dropout=self.dropout,
                         edge_dropout=self.edge_dropout,
                         is_input_layer=True
                         )

    def build_hidden_layer(self, idx):
        return RGCNLayer(self.emb_dim,
                         self.emb_dim,
                         # self.basis_weights,
                         self.aggregator,
                         attn_rel_emb_dim=self.attn_rel_emb_dim,
                         num_rels=self.num_rels,
                         rel2id=self.rel2id,
                         device=self.device,
                         activation=F.relu,
                         dropout=self.dropout,
                         edge_dropout=self.edge_dropout
                         )

    def forward(self, g):
        for layer in self.layers:
            layer(g, self.attn_rel_emb)
        return {
            "drug": g.nodes['drug'].data['repr'],
            "target": g.nodes['target'].data['repr']
        }
        # return g.ndata.pop('h')
        # return g.ndata['h']
