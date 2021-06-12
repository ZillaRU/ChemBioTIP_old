import torch
import torch.nn as nn
import torch.nn.functional as F

# reference: GraIL
from model.inter_gnn import SumAggregator, MLPAggregator, GRUAggregator
from model.inter_gnn.grail_rgcn_layer import RGCNLayer


class InterView_RGCN(nn.Module):
    def __init__(self, params):
        super(InterView_RGCN, self).__init__()
        self.inp_dim = 200,  # params.inp_dim
        self.emb_dim = 200,  # params.emb_dim
        self.attn_rel_emb_dim = 32,  # params.attn_rel_emb_dim
        self.num_rels = 3,  # params.num_rels
        self.aug_num_rels = 3,  # params.aug_num_rels
        self.num_bases = 4,  # params.num_bases
        self.num_hidden_layers = 2,  # params.num_gcn_layers
        self.dropout = 0.,  # params.dropout
        self.edge_dropout = 0.,  # params.edge_dropout
        # self.aggregator_type = params.gnn_agg_type
        self.has_attn = True,  # params.has_attn
        self.device = torch.device('cpu')  # params.device
        # print(type(3), type(self.num_rels), type(self.attn_rel_emb_dim))
        if self.has_attn:
            # self.attn_rel_emb = nn.Embedding(self.num_rels, self.attn_rel_emb_dim, sparse=False)
            self.attn_rel_emb = nn.Embedding(3, 32, sparse=False)
        else:
            self.attn_rel_emb = None
        # todo: delete
        params.gnn_agg_type = "sum"
        # initialize aggregators for input and hidden layers
        if params.gnn_agg_type == "sum":
            self.aggregator = SumAggregator(self.emb_dim)
        elif params.gnn_agg_type == "mlp":
            self.aggregator = MLPAggregator(self.emb_dim)
        elif params.gnn_agg_type == "gru":
            self.aggregator = GRUAggregator(self.emb_dim)

        # initialize basis weights for input and hidden layers
        # self.input_basis_weights = nn.Parameter(torch.Tensor(self.num_bases, self.inp_dim, self.emb_dim))
        # self.basis_weights = nn.Parameter(torch.Tensor(self.num_bases, self.emb_dim, self.emb_dim))

        # create rgcn layers
        self.build_model()

        # create initial features
        self.features = self.create_features()

    def create_features(self):  # todo: 200 -> self.inp_dim
        features = torch.arange(200).to(device=self.device)
        return features

    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
        # h2h
        for idx in range(2 - 1):  # todo: 2->self.num_hidden_layers
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)

    def build_input_layer(self):
        return RGCNLayer(self.inp_dim,
                         self.emb_dim,
                         # self.input_basis_weights,
                         self.aggregator,
                         activation=F.relu,
                         dropout=self.dropout,
                         edge_dropout=self.edge_dropout,
                         is_input_layer=True)

    def build_hidden_layer(self, idx):
        return RGCNLayer(self.emb_dim,
                         self.emb_dim,
                         # self.basis_weights,
                         self.aggregator,
                         activation=F.relu,
                         dropout=self.dropout,
                         edge_dropout=self.edge_dropout,
                         )

    def forward(self, g):
        for layer in self.layers:
            layer(g, self.attn_rel_emb)
        return g.ndata.pop('h')
