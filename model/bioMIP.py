import logging

import dgl
import torch
from torch import nn

from model.inter_GNN import InterView_RGCN
from model.intra_GNN import Intra_AttentiveFP, Intra_GAT, Intra_GCN
from model.predictors import FinalPredictor, IntraPredictor


def initialize_BioMIP(params):
    if params.load_model:  # todo
        return None
    else:
        print("============= Initialize a new BioMIP model =============")
        return BioMIP(
            params
        )


def intra_view_imputation(intra_feats, inter_adjs):
    pass


class BioMIP(nn.Module):
    """
    Args:
        --------- intra-view GNN settings --------------------------------
        featurizer: {"AttentiveFP", "Basic"}
        model_name: {"AttentiveFP", "GAT"}
        input_feat_size
        output_fear_size: default 200
        readout_type:
        layer_num: default 2

        --------- from intra-view to inter-view --------------------------
        init_inter_H0_with_intra_feat: bool, default true
        init_small_mol_with_descriptor: bool, default false

        --------- inter-view GNN settings --------------------------------
        model_name: {"HGT", "RGCN"}
        input_feat_size
        layer_num: default 2

        --------- fusion settings ----------------------------------------

        --------- predictor settings -------------------------------------

    """

    def __init__(self, params):
        super().__init__()
        # Create two sets of intra-GNNs for small- and macro-molecules respectively
        self.small_intra_gnn = Intra_AttentiveFP(
            node_feat_size=200,  # params.atom_insize
            edge_feat_size=200  # params.bond_insize
        )
        # self.macro_intra_gnn = Intra_GAT()
        self.macro_intra_gnn = Intra_AttentiveFP(
            node_feat_size=200,  # params.aa_node_insize
            edge_feat_size=1,  # params.aa_edge_insize
        )
        self.init_inter_with_intra = False  # params.init_inter_with_intra
        # Create a stack of inter-GNNs
        self.inter_gnn = InterView_RGCN(params)
        # Create 2 predictors (intra-only predictor, mixed predictor)
        self.pred1 = IntraPredictor()
        self.pred2 = FinalPredictor()

    def forward(self, small_mol_graphs, macro_mol_graphs,
                inter_graph, inter_adjs,
                small_mol_id_dict: dict, macro_mol_id_dict: dict):
        """
        Args:
            small_mol_graphs: list of DGL graphs
            macro_mol_graphs: list of DGL graphs
            inter_graph: a DGL hetero graph
            xxxxx_mol_id_dict: (key, val), mol_index in small/macro_mol_graphs --> node index in inter_graph

        Returns:

        """
        # todo:
        #  pass the small mol graph to small_intra_gnn
        small_mol_feats = self.small_intra_gnn(small_mol_graphs)
        macro_mol_feats = self.macro_intra_gnn(macro_mol_graphs)

        # note that there are many nodes in inter-view graph without corresponding intra-view features
        valid_small_mol_num, valid_macro_mol_num = small_mol_feats.shape[0], macro_mol_feats.shape[0]
        num_nodes = inter_graph.num_nodes()
        small_missing_rate = valid_small_mol_num / inter_graph.num_nodes('small')
        macro_missing_rate = valid_macro_mol_num / inter_graph.num_nodes('macro')
        total_missing_rate = (valid_small_mol_num + valid_macro_mol_num) / num_nodes
        logging.info(f"small molecule missing rate: {small_missing_rate}\n" +
                     f"macro molecule missing rate: {macro_missing_rate}\n" +
                     f"total missing rate: {total_missing_rate}")

        # mapping small_mol_feats and macro_mol_feats with corresponding nodes in the inter-graph --> intra_mol_feats
        # todo: parameterize "200" intra_out_size == inter_in_size
        intra_mol_feats = torch.zeros((num_nodes, 200))
        for (k, v) in small_mol_id_dict.items():
            intra_mol_feats[v] = small_mol_feats[k]
        for (k, v) in macro_mol_id_dict.items():
            intra_mol_feats[v] = macro_mol_feats[k]

        # inter-connectivity-guided data imputation for missing intra-view
        intra_view_imputation(intra_mol_feats, inter_adjs)

        # Inter GNN
        # pass the macro mol graph to macro_intra_gnn, use the output of intra-GNN to initialize H0 in 0th layer of
        # inter-GNN heterograph ndata['type']['feat']
        # todo: init with labeling
        inter_graph.ndata['hv'] = intra_mol_feats if self.init_inter_with_intra else torch.zeros((num_nodes, 200))

        # inter_graph.edata['he'] = None
        inter_mol_feats = self.inter_gnn(inter_graph)

        # combine the output features from intra-GNNs and inter-GNNs, and predict
        return self.pred1(intra_mol_feats, inter_mol_feats), self.pred2(intra_mol_feats, inter_mol_feats)


def collate(graphs):
    return dgl.batch(graphs)
