import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from scipy.special import softmax
from tqdm import tqdm

from utils.graph_utils import get_edge_count


def process_files(pos_file, saved_relation2id=None):
    '''
    files: Dictionary map of file paths to read the triplets from.
    saved_relation2id: Saved relation2id (mostly passed from a trained model) which can be used to map relations to pre-defined indices and filter out the unknown ones.
    '''
    entity2id = {}
    relation2id = {} if saved_relation2id is None else saved_relation2id

    ent = 0
    rel = 0

    data = []
    with open(pos_file) as f:
        file_data = [line.split(',') for line in f.read().split('\n')[:-1]]

    for triplet in file_data:
        if triplet[0] not in entity2id:
            entity2id[triplet[0]] = ent
            ent += 1
        if triplet[2] not in entity2id:
            entity2id[triplet[2]] = ent
            ent += 1
        if not saved_relation2id and triplet[1] not in relation2id:
            relation2id[triplet[1]] = rel
            rel += 1

        # Save the triplets corresponding to only the known relations
        if triplet[1] in relation2id:
            data.append([entity2id[triplet[0]], entity2id[triplet[2]], relation2id[triplet[1]]])

    triplets = np.array(data)

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    # Construct the list of adjacency matrix each corresponding to eeach relation. Note that this is constructed only from the train data.
    adj_list = []
    for i in range(len(relation2id)):
        idx = np.argwhere(triplets[:, 2] == i)
        adj_list.append(csc_matrix(
            (np.ones(len(idx), dtype=np.uint8), (triplets[:, 0][idx].squeeze(1), triplets[:, 1][idx].squeeze(1))),
            shape=(len(entity2id), len(entity2id))))

    return adj_list, triplets, entity2id, relation2id, id2entity, id2relation


def sample_neg(adj_list, edges, num_neg_samples_per_link=1, max_size=1000000, constrained_neg_prob=0):
    pos_edges = edges
    neg_edges = []

    # if max_size is set, randomly sample train links
    if max_size < len(pos_edges):
        perm = np.random.permutation(len(pos_edges))[:max_size]
        pos_edges = pos_edges[perm]

    # sample negative links for train/test
    n, r = adj_list[0].shape[0], len(adj_list)

    # distribution of edges across reelations
    theta = 0.001
    edge_count = get_edge_count(adj_list)
    rel_dist = np.zeros(edge_count.shape)
    idx = np.nonzero(edge_count)
    rel_dist[idx] = softmax(theta * edge_count[idx])

    # possible head and tails for each relation
    valid_heads = [adj.tocoo().row.tolist() for adj in adj_list]
    valid_tails = [adj.tocoo().col.tolist() for adj in adj_list]

    pbar = tqdm(total=len(pos_edges))
    while len(neg_edges) < num_neg_samples_per_link * len(pos_edges):
        neg_head, neg_tail, rel = pos_edges[pbar.n % len(pos_edges)][0], pos_edges[pbar.n % len(pos_edges)][1], \
                                  pos_edges[pbar.n % len(pos_edges)][2]
        if np.random.uniform() < constrained_neg_prob:
            if np.random.uniform() < 0.5:
                neg_head = np.random.choice(valid_heads[rel])
            else:
                neg_tail = np.random.choice(valid_tails[rel])
        else:
            if np.random.uniform() < 0.5:
                neg_head = np.random.choice(n)
            else:
                neg_tail = np.random.choice(n)

        if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
            neg_edges.append([neg_head, neg_tail, rel])
            pbar.update(1)

    pbar.close()
    neg_edges = np.array(neg_edges)
    return neg_edges


adj_list, triplets, entity2id, relation2id, id2entity, id2relation = process_files(
    "/Users/zilla/PycharmProjects/BioMIP/data/default_drugbank/dti_pos.csv")

print(id2relation)

neg_edges = sample_neg(adj_list,
                       triplets,
                       num_neg_samples_per_link=1,
                       ).tolist()
# print(neg_edges.shape)
for i in range(len(neg_edges)):
    uvr = neg_edges[i]
    u, r, v = id2entity[uvr[0]], id2relation[uvr[2]], id2entity[uvr[1]]
    neg_edges[i] = [u, r, v]

pd.DataFrame(neg_edges).to_csv('/Users/zilla/PycharmProjects/BioMIP/data/default_drugbank/valid_dti_neg_rate1.csv',
                               header=None,
                               index=False)
