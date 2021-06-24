import logging
import os
import pandas as pd
import dgl
import torch

from model.bioMIP import initialize_BioMIP
from sample_batch import NegativeSampler, get_dgl_minibatch_dataloader
from utils.arg_parser import parser
from utils.data_utils import build_inter_graph_from_links, ssp_multigraph_to_dgl
from utils.generate_intra_graph_db import generate_small_mol_graph_datasets, generate_macro_mol_graph_datasets
from utils.intra_graph_dataset import IntraGraphDataset
# from utils.subgraph_dataset import generate_subgraph_datasets


def train_epoch_full_batch(model, small_mol_graphs, macro_mol_graphs, inter_graph, params):
    pass


def train_epoch_mini_batch(model,
                           small_set, macro_set,
                           small_mol_graphs, macro_mol_graphs,
                           inter_graph,
                           id2entity,
                           sampler='dgl'
                           ):
    if sampler == 'dgl':
        dataloader = get_dgl_minibatch_dataloader(inter_graph)
    else:
        pass
    # print(inter_graph, id2entity)

    for input_nodes, positive_graph, negative_graph, blocks in dataloader:
        """
        positive_graph 包含采样得到的小批次内所有的边。
        negative_graph 包含由负采样方法生成的所有不存在的边。
        blocks 邻居采样方法生成的块的列表。
        """
        # print(input_nodes)  # tensor([2, 0, 3]) == positive_graph.ndata['_ID']
        # map node_id in full_inter_graph to tp
        blocks = [b for b in blocks]
        # positive_graph = positive_graph.to(torch.device('cuda'))
        # negative_graph = negative_graph.to(torch.device('cuda'))
        # input_features = blocks[0].srcdata['features']
        # small_mol_graphs
        batch_small_graphs, batch_macro_graphs = {}, {}
        print(id2entity)
        for i in input_nodes.tolist():
            identifier = id2entity[i]
            if identifier in small_set:
                _, batch_small_graphs[i], _ = small_mol_graphs[identifier]
            elif identifier in macro_set:
                _, _, batch_macro_graphs[i] = macro_mol_graphs[identifier]
            else:
                pass
        # Here the mol id in batch_xxxxx_graphs is same with the mol in the full inter_graph
        pos_score, neg_score = model(batch_small_graphs, batch_macro_graphs,
                                     positive_graph, negative_graph,
                                     blocks)
        loss = compute_loss(pos_score, neg_score)
        # opt.zero_grad()
        # loss.backward()
        # opt.step()


def train_BioMIP(n_epoch, model, small_set, macro_set, small_mol_graphs, macro_mol_graphs, inter_graph, id2entity):
    # loss =
    for i in range(1, n_epoch + 1):
        train_epoch_mini_batch(model, small_set, macro_set, small_mol_graphs, macro_mol_graphs, inter_graph, id2entity)
        # train_epoch_full_batch(model, small_mol_graphs, macro_mol_graphs, inter_graph, params)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    params = parser.parse_args()
    # initialize_experiment(params)
    # save featurized intra-view graphs
    print(params)
    params.small_mol_db_path = f'data/{params.dataset}/smile_graph_db_{params.SMILES_featurizer}'
    params.macro_mol_db_path = f'data/{params.dataset}/prot_graph_db'  # _{params.prot_featurizer}

    # load the list of molecules with available chemical structures
    small_mol_list = pd.read_csv(f'data/{params.dataset}/SMILESstrings.csv', header=None, names=['id', '_'])['id'].tolist()
    macro_mol_list = pd.read_csv(f'data/{params.dataset}/macro_seqs.csv', header=None, names=['id', '_'])['id'].tolist()
    small_mol_list = [str(i) for i in small_mol_list]
    macro_mol_list = [str(i) for i in macro_mol_list]

    print('small molecule db_path:', params.small_mol_db_path)
    print('macro molecule db_path:', params.macro_mol_db_path)

    # if not processed, build intra-view graphs
    if not os.path.isdir(params.small_mol_db_path):
        generate_small_mol_graph_datasets(params)
    if not os.path.isdir(params.macro_mol_db_path):
        generate_macro_mol_graph_datasets(params)

    # load intra-view graph dataset
    small_mol_graphs = IntraGraphDataset(db_path=params.small_mol_db_path, db_name='small_mol')
    macro_mol_graphs = IntraGraphDataset(db_path=params.macro_mol_db_path, db_name='macro_mol')

    params.atom_insize = small_mol_graphs.get_nfeat_dim()
    params.bond_insize = small_mol_graphs.get_efeat_dim()
    params.aa_node_insize = macro_mol_graphs.get_nfeat_dim()
    params.aa_edge_insize = macro_mol_graphs.get_efeat_dim()
    # init model and loss
    model = initialize_BioMIP(params)

    # load the inter-view graph
    adj_list_dict, triplets, \
    entity2id, relation2id, \
    id2entity, id2relation, id_to_type = build_inter_graph_from_links(
        params.dataset,
        {
            'train': f'data/{params.dataset}/train.txt',
            'valid': f'data/{params.dataset}/valid.txt'
        })

    inter_graph = ssp_multigraph_to_dgl(
        dataset='mini',
        graphs=adj_list_dict
        # n_feats=None
    )
    print(macro_mol_list)
    train_BioMIP(params.n_epoch,
                 model,
                 set(small_mol_list), set(macro_mol_list),
                 small_mol_graphs,
                 macro_mol_graphs,
                 inter_graph,
                 id2entity)
