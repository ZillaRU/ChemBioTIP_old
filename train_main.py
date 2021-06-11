import os
import logging
from utils.arg_parser import parser
import torch

from utils.generate_intra_graph_db import generate_small_mol_graph_datasets, generate_macro_mol_graph_datasets
from utils.intra_graph_dataset import IntraGraphDataset

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    params = parser.parse_args()
    # initialize_experiment(params)
    # save featurized intra-view graphs
    print(params)
    params.small_mol_db_path = f'data/{params.dataset}/smile_graph_db_{params.SMILES_featurizer}'
    params.macro_mol_db_path = f'data/{params.dataset}/prot_graph_db'  # _{params.prot_featurizer}
    print('small molecule db_path:', params.small_mol_db_path)
    print('macro molecule db_path:', params.macro_mol_db_path)
    if not os.path.isdir(params.small_mol_db_path):
        generate_small_mol_graph_datasets(params)
    if not os.path.isdir(params.macro_mol_db_path):
        generate_macro_mol_graph_datasets(params)

    small_mol_graphs = IntraGraphDataset(db_path=params.small_mol_db_path, db_name='small_mol')
    macro_mol_graphs = IntraGraphDataset(db_path=params.macro_mol_db_path, db_name='macro_mol')
