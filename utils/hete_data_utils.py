import dgl
import numpy as np
import torch
from scipy.sparse import csc_matrix


def build_inter_graph_from_links(dataset, saved_relation2id=None):
    files = {
        'train': {
            'pos': f'data/{dataset}/train_pos.txt',
            'neg': f'data/{dataset}/train_neg.txt'
        },
        'valid': {
            'pos': f'data/{dataset}/valid_pos.txt',
            'neg': f'data/{dataset}/train_neg.txt'
        }
    }
    if dataset == 'default_drugbank':
        # biodrug_list = list(pd.read_csv(f'data/{dataset}/drug_seqs.csv', header=None).iloc[:, 0])
        dt_types = {'targets', 'enzymes', 'carriers', 'transporters'}
        tt_types = {}
    else:
        dt_types = {'dt'}
        tt_types = {}

    drug2id, target2id = {}, {}
    relation2id = {} if not saved_relation2id else None
    drug_cnt, target_cnt = 0, 0
    rel = 0
    triplets = {}

    for file_type, file_paths in files.items():
        triplets[file_type] = {}
        for y, path in file_paths.items():
            data = []
            with open(path) as f:
                file_data = [line.split(',') for line in f.read().split('\n')[:-1]]
            for [u, r, v] in file_data:
                if r == 'dt':
                    u_is_d, v_is_d = True, False
                else:
                    u_is_d, v_is_d = u.startswith('DB'), v.startswith('DB')
                if u_is_d and u not in drug2id:
                    drug2id[u] = drug_cnt
                    drug_cnt += 1
                if not u_is_d and u not in target2id:
                    target2id[u] = target_cnt
                    target_cnt += 1
                if v_is_d and v not in drug2id:
                    drug2id[v] = drug_cnt
                    drug_cnt += 1
                if not v_is_d and v not in target2id:
                    target2id[v] = target_cnt
                    target_cnt += 1
                if not saved_relation2id and r not in relation2id:
                    relation2id[r] = rel
                    rel += 1
                # Save the triplets corresponding to only the known relations
                if r in relation2id:
                    data.append([drug2id[u] if u_is_d else target2id[u],
                                 drug2id[v] if v_is_d else target2id[v],
                                 relation2id[r]])
            triplets[file_type][y] = np.array(data, dtype=np.uint16)

    id2drug = {v: k for k, v in drug2id.items()}
    id2target = {v: k for k, v in target2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    # Construct the list of adjacency matrix each corresponding to each relation.
    # Note that this is constructed only from the train data.
    adj_dict = {}
    for i in range(len(relation2id)):
        idx = np.argwhere(triplets['train']['pos'][:, 2] == i)
        rel = id2relation[i]
        rel_tuple = (
            "target" if (rel in tt_types or rel in dt_types) else "drug",
            rel,
            "drug" if rel not in tt_types else "target"
        ) if rel != 'dt' else ('drug', 'dt', 'target')
        shape = (
            target_cnt if (rel in tt_types or rel in dt_types) else drug_cnt,
            drug_cnt if rel not in tt_types else target_cnt
        )
        if rel == 'dt':
            shape = (drug_cnt, target_cnt)
        print(rel, shape)
        adj_dict[rel_tuple] = csc_matrix(
            (
                np.ones(len(idx), dtype=np.uint8),
                (
                    triplets['train']['pos'][:, 0][idx].squeeze(1),
                    triplets['train']['pos'][:, 1][idx].squeeze(1)
                )
            ), shape=shape)
    print(drug_cnt, target_cnt)
    return adj_dict, triplets, \
           drug2id, target2id, relation2id, \
           id2drug, id2target, id2relation


def ssp_multigraph_to_dgl(adjs, relation2id):
    adjs = {k: v.tocoo() for k, v in adjs.items()}
    # g_dgl = dgl.heterograph({
    #     k: (torch.from_numpy(v.row), torch.from_numpy(v.col)) for k, v in adjs.items()
    # })
    # return dgl.to_bidirected(g_dgl)
    graph_dict = {}
    for k, v in adjs.items():
        print(k)
        if k[0] != k[2]:
            graph_dict[k] = (torch.from_numpy(v.row), torch.from_numpy(v.col))
            graph_dict[(k[2], f"~{k[1]}", k[0])] = (torch.from_numpy(v.col), torch.from_numpy(v.row))
            relation2id[f"~{k[1]}"] = len(relation2id)
        else:
            # graph_dict[k] = (torch.from_numpy(np.hstack((v.row, v.col))),
            #                  torch.from_numpy(np.hstack((v.col, v.row))))
            graph_dict[k] = (torch.from_numpy(v.row),
                             torch.from_numpy(v.col))
    # g_dgl = dgl.heterograph({
    #     k: (torch.from_numpy(np.hstack((v.row, v.col))),
    #         torch.from_numpy(np.hstack((v.col, v.row))))
    #     for k, v in adjs.items()
    # })
    g_dgl = dgl.heterograph(graph_dict)
    return g_dgl, relation2id
