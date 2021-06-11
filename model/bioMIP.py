

class bioMIP:
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