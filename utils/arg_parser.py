import argparse

parser = argparse.ArgumentParser(description='TransE model)')
# Experiment setup params
parser.add_argument("--experiment_name", "-e", type=str, default="default",
                    help="A folder with this name would be created to dump saved models and log files")
parser.add_argument("--dataset", "-d", type=str,
                    default='mini',
                    help="Dataset string")
parser.add_argument("--gpu", type=int, default=0,
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