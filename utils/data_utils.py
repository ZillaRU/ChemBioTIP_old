# import pickle
import os

import dgl
import dill
import networkx as nx
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy.sparse import csc_matrix


def serialize(data):
    data_tuple = tuple(data.values())
    return dill.dumps(data_tuple)


# def deserialize_small(data):
#     keys = ('mol_id', 'mol_graph', 'graph_size')
#     return dict(zip(keys, dill.loads(data)))


def deserialize_small(data):
    keys = ('mol_graph', 'graph_size')
    return dict(zip(keys, dill.loads(data)))


def deserialize_macro(data):
    keys = ('seq', 'mol_graph')
    dill.loads(data)
    return dict(zip(keys, dill.loads(data)))


# def deserialize_macro(data):
#     keys = ('mol_id', 'seq', 'mol_graph')
#     dill.loads(data)
#     return dict(zip(keys, dill.loa ds(data)))

# todo
def deserialize_subgraph(data):
    keys = ('mol_id', 'seq', 'mol_graph')
    dill.loads(data)
    return dict(zip(keys, dill.loads(data)))


# todo
def identify_type_drugbank(name: str, _refer):
    if str == 'DB00016':
        return 'macro' if str in _refer else 'small'


def identify_type(name: str, _refer):
    pass


def plot_rel_dist(adj_list, filename):
    rel_count = []
    for adj in adj_list:
        rel_count.append(adj.count_nonzero())

    fig = plt.figure(figsize=(12, 8))
    plt.plot(rel_count)
    fig.savefig(filename, dpi=fig.dpi)


# triplet format: (u, rt, v)
def build_inter_graph_from_links(dataset, files: dict, saved_relation2id: bool = False):
    if dataset == 'drugbank':
        biodrug_list = pd.read_csv('../data/drugbank/biodrugs.csv', header=None).to_list()
        type_marker = identify_type_drugbank
        _refer = set(biodrug_list)
    else:
        biodrug_list = ['DB00016']
        type_marker = identify_type_drugbank
        _refer = set(biodrug_list)
    entity2id = {}
    relation2id = {} if not saved_relation2id else None
    id2type = {}
    ent = 0
    rel = 0
    triplets = {}

    for file_type, file_path in files.items():
        data = []
        with open(file_path) as f:
            file_data = [line.split() for line in f.read().split('\n')[:-1]]
        for triplet in file_data:
            if triplet[0] not in entity2id:
                entity2id[triplet[0]] = ent
                id2type[ent] = type_marker(triplet[0], _refer)
                ent += 1
            if triplet[2] not in entity2id:
                entity2id[triplet[2]] = ent
                id2type[ent] = type_marker(triplet[2], _refer)
                ent += 1
            if not saved_relation2id and triplet[1] not in relation2id:
                relation2id[triplet[1]] = rel
                rel += 1

            # Save the triplets corresponding to only the known relations
            if triplet[1] in relation2id:
                data.append([entity2id[triplet[0]], entity2id[triplet[2]], relation2id[triplet[1]]])

        triplets[file_type] = np.array(data)

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    # Construct the list of adjacency matrix each corresponding to eeach relation. Note that this is constructed only from the train data.
    adj_list = dict()
    for i in range(len(relation2id)):
        idx = np.argwhere(triplets['train'][:, 2] == i)
        adj_list[id2relation[i]] = csc_matrix(
            (
                np.ones(len(idx), dtype=np.uint8),
                (
                    triplets['train'][:, 0][idx].squeeze(1),
                    triplets['train'][:, 1][idx].squeeze(1)
                )
            ), shape=(len(entity2id), len(entity2id)))
    return adj_list, triplets, entity2id, relation2id, id2entity, id2relation, id2type


def ssp_multigraph_to_dgl(dataset, graphs, n_feats=None):
    """
    Converting ssp multigraph (i.e. list of adjs) to dgl multigraph.
    """
    if dataset == 'mini':
        mini_meta_rel = {
            'rel3': ('drug', 'rel3', 'drug'),
            'rel2': ('drug', 'rel2', 'drug'),
            '-': ('drug', '-', 'drug')
        }
        print(graphs)
        graph_data = {
            mini_meta_rel[rel]: (list(zip(adj.tocoo().row, adj.tocoo().col)))
            for (rel, adj) in graphs.items()
        }
        g_dgl = dgl.heterograph(graph_data)
        # if n_feats is not None:
        #     g_dgl.ndata['drug']['feat'] = torch.tensor(n_feats)
        return g_dgl


def save_to_file(directory, file_name, triplets, id2entity, id2relation):
    file_path = os.path.join(directory, file_name)
    with open(file_path, "w") as f:
        for s, o, r in triplets:
            f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o]]) + '\n')
