import argparse

parser = argparse.ArgumentParser(description='TransE model)')
# Experiment setup params
parser.add_argument("--experiment_name", "-e", type=str, default="default",
                    help="A folder with this name would be created to dump saved models and log files")
parser.add_argument("--dataset", "-d", type=str,
                    default='default_drugbank',
                    help="Dataset string")
parser.add_argument("--gpu", type=int, default=-1,
                    help="Which GPU to use?")
parser.add_argument('--disable_cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--load_model', action='store_true',
                    help='Load existing model?')
parser.add_argument("--train_file", "-tf", type=str, default="train",
                    help="Name of file containing training triplets")
parser.add_argument("--valid_file", "-vf", type=str, default="valid",
                    help="Name of file containing validation triplets")

parser.add_argument("--SMILES_featurizer", "-smfeat", type=str, default='base')

parser.add_argument("--intra_out_dim", type=int, default=200,
                    help="the feature dimension of molecules from intra-view")

parser.add_argument("--inp_dim", type=int, default=200,
                    help="the input feature dimension of molecules from inter-view, and it should be consistent with the intra_out_dim")

parser.add_argument("--emb_dim", type=int, default=200,
                    help="the out feature dimension of molecules from inter-view, and it should be consistent with the inp_dim")

parser.add_argument("--mode", type=str, choices=['hard_mix', 'soft_mix', 'two_pred'], default='hard_mix',
                    help="predictor mode")

parser.add_argument("--rel_emb_dim", type=int, default=32)

parser.add_argument("--num_rels", type=int)

parser.add_argument("--task", type=str, default='ddi', choices=['ddi', 'dta'])

parser.add_argument("--target", type=str, default='01', choices=['01', '0_1'],
                    help="classification or regression")

parser.add_argument('--n_epoch', type=int, default=100)
