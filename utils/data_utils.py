# import pickle

import dgl
import dill
import networkx as nx
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csc_matrix


def serialize(data):
    data_tuple = tuple(data.values())
    return dill.dumps(data_tuple)


def deserialize_small(data):
    keys = ('mol_id', 'mol_graph', 'graph_size')
    return dict(zip(keys, dill.loads(data)))


def deserialize_macro(data):
    keys = ('mol_id', 'seq', 'mol_graph')
    dill.loads(data)
    return dict(zip(keys, dill.loads(data)))


# todo
def identify_type_drugbank(name: str, _refer):
    if str == 'DB00016':
        return 'macro' if str in _refer else 'small'


def identify_type(name: str, _refer):
    pass


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
    adj_list = []
    for i in range(len(relation2id)):
        idx = np.argwhere(triplets['train'][:, 2] == i)
        adj_list.append(csc_matrix(
            (
                np.ones(len(idx), dtype=np.uint8),
                (
                    triplets['train'][:, 0][idx].squeeze(1),
                    triplets['train'][:, 1][idx].squeeze(1)
                )
            ), shape=(len(entity2id), len(entity2id))))
    return adj_list, triplets, entity2id, relation2id, id2entity, id2relation, id2type


def ssp_multigraph_to_dgl(graph, n_feats=None):
    """
    Converting ssp multigraph (i.e. list of adjs) to dgl multigraph.
    """

    g_nx = nx.MultiDiGraph()
    g_nx.add_nodes_from(list(range(graph[0].shape[0])))
    # Add edges
    for rel, adj in enumerate(graph):
        # Convert adjacency matrix to tuples for nx0
        nx_triplets = []
        for src, dst in list(zip(adj.tocoo().row, adj.tocoo().col)):
            nx_triplets.append((src, dst, {'type': rel}))
        g_nx.add_edges_from(nx_triplets)

    # make dgl graph
    # g_dgl = dgl.graph(g_nx)
    # dgl.from_scipy()
    g_dgl = dgl.from_networkx(g_nx, node_attrs=[], edge_attrs=['type'])
    # add node features
    if n_feats is not None:
        g_dgl.ndata['feat'] = torch.tensor(n_feats)
    return g_dgl
