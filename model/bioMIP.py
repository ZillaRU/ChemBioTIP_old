import dgl
import torch
from torch import nn

from model.inter_GNN import InterView_RGCN
from model.intra_GNN import Intra_AttentiveFP
from model.predictors import FinalPredictor, IntraPredictor, HardMixPredictor, SoftMixPredictor
from model_full.inter_gnn import RGCN


def initialize_BioMIP(params):
    if params.load_model:
        return None
    else:
        print("============= Initialize a new BioMIP model =============")
        return BioMIP(
            params
        )


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
        self.params = params
        self.device = params.device
        # Create two sets of intra-GNNs for small- and macro-molecules respectively
        self.small_intra_gnn = Intra_AttentiveFP(
            node_feat_size=params.atom_insize,
            edge_feat_size=params.bond_insize,
            graph_feat_size=params.intra_out_dim,
            num_layers=params.intra_small_layer_num,
            dropout=params.intra_small_dropout
        ).to(self.device)
        # self.macro_intra_gnn = Intra_GAT()
        self.macro_intra_gnn = Intra_AttentiveFP(
            node_feat_size=params.aa_node_insize,
            edge_feat_size=params.aa_edge_insize,
            graph_feat_size=params.intra_out_dim,
            num_layers=params.intra_macro_layer_num,
            dropout=params.intra_macro_dropout
        ).to(self.device)
        self.init_inter_with_intra = False  # params.init_inter_with_intra
        # Create a stack of inter-GNNs
        params.inter_gnn = 'default'
        if params.inter_gnn == 'grail_rgcn':
            self.inter_gnn = InterView_RGCN(
                inp_dim=params.intra_out_dim,
                emb_dim=params.inter_emb_dim,
                attn_rel_emb_dim=params.attn_rel_emb_dim,
                num_rels=params.num_rels,
                rel2id=params.rel2id,
                aug_num_rels=params.num_rels,
                num_bases=params.num_bases,  # 基数分解
                num_gcn_layers=params.inter_layer_num,
                dropout=params.inter_dropout,
                edge_dropout=params.inter_edge_dropout,
                gnn_agg_type=params.inter_agg,
                has_attn=params.inter_has_attn,
                device=params.device
            ).to(self.device)
        else:
            # self.inter_gnn = HeteroRGCN
            self.inter_gnn = RGCN(params.intra_out_dim,
                                  params.inter_emb_dim,
                                  params.inter_emb_dim,
                                  rel_names=list(params.rel2id.keys())
                                  ).to(self.device)
        # self.rel_emb1 = nn.Embedding(num_embeddings=params.num_rels,
        #                              embedding_dim=params.rel_emb_dim)  # for negative sampling

        # Create predictor (intra-only predictor, mixed predictor)
        self.mode = params.mode
        if self.mode == 'two_pred':
            self.pred1 = IntraPredictor(params.num_rels, params.rel_emb_dim)
            self.pred2 = FinalPredictor(params.num_rels, params.rel_emb_dim)
        elif self.mode == 'soft_mix':
            self.pred = SoftMixPredictor()
        else:  # elif params.mode == 'hard_mix':
            self.pred = HardMixPredictor(params.num_rels,
                                         params.id2rel,
                                         params.rel_emb_dim,
                                         self.device,
                                         intra_fc_in=params.intra_out_dim,
                                         inter_fc_in=params.emb_dim)  # * params.inter_layer_num)  # 2 layer_num

    def forward(self,
                curr_smalls,
                curr_biotechs,
                curr_targets,
                nid_drugs,
                nid_targets,
                inter_graph,
                triplets
                ):
        #  pass the small mol graph to small_intra_gnn
        small_bg = dgl.batch([i[1] for i in list(curr_smalls.values())]).to(device=self.device)
        # drugbank only
        # biotech_bg = dgl.batch([i[2] for i in list(curr_biotechs.values())]).to(device=self.device)
        target_bg = dgl.batch([i[2] for i in list(curr_targets.values())]).to(device=self.device)
        small_mol_feats = self.small_intra_gnn(small_bg,
                                               small_bg.ndata['nfeats'].to(torch.float32).to(device=self.device),
                                               small_bg.edata['efeats'].to(torch.float32).to(device=self.device)).to(
            device=self.device)
        # drugbank only
        # biotech_feats = self.macro_intra_gnn(biotech_bg,
        #                                      biotech_bg.ndata['nfeats'].to(torch.float32).to(device=self.device),
        #                                      biotech_bg.edata['efeats'].to(torch.float32).to(device=self.device)).to(
        #     device=self.device)
        target_feats = self.macro_intra_gnn(target_bg,
                                            target_bg.ndata['nfeats'].to(torch.float32).to(device=self.device),
                                            target_bg.edata['efeats'].to(torch.float32)).to(device=self.device).to(
            device=self.device)
        num_small, num_target = small_mol_feats.shape[0], target_feats.shape[0]
        # drugbank only
        # num_biotech = biotech_feats.shape[0],
        num_drug, num_target = inter_graph.num_nodes('drug'), inter_graph.num_nodes('target')
        # assert num_small + num_biotech == num_drug

        # mapping small_mol_feats and macro_mol_feats with corresponding nodes in the inter-graph --> intra_mol_feats
        intra_drug_feats = torch.zeros((num_drug, self.params.intra_out_dim)).to(device=self.device)
        # print(list(curr_smalls.keys()))
        # print(list(curr_biotechs.keys()))
        # print(nid_drugs)
        # print(list(curr_targets.keys()))
        # print(nid_targets)

        _i, s_i, b_i = 0, 0, 0
        drug_id2subid = {}
        for id in nid_drugs:
            if id in curr_smalls:
                intra_drug_feats[_i] = small_mol_feats[s_i]
                s_i += 1
            # else: # drugbank only
            # intra_drug_feats[_i] = biotech_feats[b_i]
            # b_i += 1
            drug_id2subid[id] = _i
            _i += 1
        # print(_i, s_i, b_i)

        target_id2subid = {}
        _i = 0
        for id in nid_targets:
            target_id2subid[id] = _i

        # print("intra_drug_feats", intra_drug_feats)

        # Inter GNN
        # pass the macro mol graph to macro_intra_gnn, use the output of intra-GNN to initialize H0 in 0th layer of
        # inter-GNN heterograph g.nodes['type'].data['feat']
        # inter_graph.nodes['drug'].data['intra'] = intra_drug_feats
        # inter_graph.nodes['target'].data['intra'] = target_feats

        # inter_graph.nodes['drug'].data['h'] = torch.clone(intra_drug_feats)
        # inter_graph.nodes['target'].data['h'] = torch.clone(target_feats)

        inter_graph.nodes['drug'].data['intra'] = intra_drug_feats
        # print("inter_graph.nodes['drug'].data['h']", inter_graph.nodes['drug'].data['h'])
        # print("intra_drug_feats", intra_drug_feats)
        inter_graph.nodes['target'].data['intra'] = target_feats

        inter_feats = self.inter_gnn(inter_graph)
        # print("inter_feats", inter_feats)
        # inter_graph.ndata['repr'] = self.inter_gnn(inter_graph)
        # print("inter_graph.ndata['repr']", inter_graph.ndata['repr'])
        # print("inter_graph.nodes['drug'].data['repr']", inter_graph.nodes['drug'].data['repr'])
        # print("inter_graph.nodes['target'].data['repr']", inter_graph.nodes['target'].data['repr'])
        # print("inter_graph.ndata['intra']", inter_graph.ndata['intra'])
        # print("inter_graph.nodes['drug'].data['intra']", inter_graph.nodes['drug'].data['intra'])
        # print("inter_graph.nodes['target'].data['intra']", inter_graph.nodes['target'].data['intra'])

        # combine the output features from intra-GNNs and inter-GNNs, and predict
        if self.mode == 'two_pred':
            return self.pred1(intra_drug_feats,
                              target_feats,
                              triplets), \
                   self.pred2(intra_drug_feats,
                              # inter_graph.nodes['drug'].data['repr'],
                              inter_feats['drug'],
                              target_feats,
                              inter_feats['target'],
                              # inter_graph.nodes['target'].data['repr'],
                              triplets)
        else:
            return self.pred(intra_drug_feats,
                             inter_feats['drug'],
                             # inter_graph.nodes['drug'].data['repr'],
                             drug_id2subid,
                             target_feats,
                             inter_feats['target'],
                             # inter_graph.nodes['target'].data['repr'],
                             target_id2subid,
                             triplets)
