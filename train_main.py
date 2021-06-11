import logging
import os

from model.bioMIP import initialize_BioMIP
from utils.arg_parser import parser
from utils.data_utils import build_inter_graph_from_links, ssp_multigraph_to_dgl
from utils.generate_intra_graph_db import generate_small_mol_graph_datasets, generate_macro_mol_graph_datasets
from utils.intra_graph_dataset import IntraGraphDataset


def train_an_epoch():
    pass


def train(model, small_mol_graphs, macro_mol_graphs, inter_graph):
    train_an_epoch()


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

    # if not processed, build intra-view graphs
    if not os.path.isdir(params.small_mol_db_path):
        generate_small_mol_graph_datasets(params)
    if not os.path.isdir(params.macro_mol_db_path):
        generate_macro_mol_graph_datasets(params)

    # load intra-view graphs
    small_mol_graphs = IntraGraphDataset(db_path=params.small_mol_db_path, db_name='small_mol')
    macro_mol_graphs = IntraGraphDataset(db_path=params.macro_mol_db_path, db_name='macro_mol')

    # load the inter-view graph

    adj_list, triplets, \
    entity2id, relation2id, \
    id2entity, id2relation, id_to_type = build_inter_graph_from_links(
        params.dataset,
        {
            'train': f'data/{params.dataset}/train.csv',
            'valid': f'data/{params.dataset}/valid.csv'
        })

    inter_graph = ssp_multigraph_to_dgl(adj_list)

    # init model and loss
    model = initialize_BioMIP(params)
    train(params.epochs, model, small_mol_graphs, macro_mol_graphs, inter_graph)
