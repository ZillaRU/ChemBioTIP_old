from torch import nn


def initialize_BioMIP(params):
    if params.load_model:
        return None
    else:
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
    def __init__(self):
        super().__init__()
        # Create two sets of intra-GNNs for small- and macro-molecules respectively

        # Create a stack of inter-GNNs

        # Create 2 predictors (intra-only predictor, mixed predictor)

    def forward(self):
        pass
        # use the output of intra-GNN to initialize H0 in 0th layer of inter-GNN

        # inter-GNN

        # combine the output features from intra-GNNs and inter-GNNs, and predict





