import logging
import os
import time

import pandas as pd
import torch
from sklearn import metrics
from torch import nn

from model.bioMIP import initialize_BioMIP
from sample_batch import sample_BioM_subgraph
from utils.arg_parser import parser
from utils.data_utils import convert_triplets_to_Xy
from utils.generate_intra_graph_db import generate_macro_mol_graph_datasets, generate_small_mol_graph_datasets
from utils.hete_data_utils import build_inter_graph_from_links, ssp_multigraph_to_dgl
from utils.intra_graph_dataset import IntraGraphDataset


# from utils.subgraph_dataset import generate_subgraph_datasets


def train_epoch_full_batch(model, small_mol_graphs, macro_mol_graphs, inter_graph, params):
    pass


# def train_on_subgraph(model,
#                       small_set, macro_set,
#                       small_mol_graphs, macro_mol_graphs,
#                       inter_graph,
#                       relation2id,
#                       triplet_ys,
#                       id2drug, id2target,
#                       etype_deg_dict,
#                       sampler='bio_m',
#                       pred_mode='two_pred'
#                       ):
#     if sampler == 'bio_m':
#         sampled_inter_graph, nids, valid_triplets_y = sample_BioM_subgraph(inter_graph, relation2id, etype_deg_dict,
#                                                                            [1., 1.],
#                                                                            triplet_ys)
#     else:
#         raise NotImplementedError
#     curr_smalls, curr_biotechs, curr_targets = {}, {}, {}
#     for id in nids['drug']:
#         DB_id = id2drug[id]
#         if DB_id in small_set:
#             curr_smalls[id] = small_mol_graphs[DB_id]
#         else:  # is biotech drug
#             # print(DB_id)
#             curr_biotechs[id] = macro_mol_graphs[DB_id]
#
#     for id in nids['target']:
#         T_id = id2target[id]
#         print(T_id)
#         curr_targets[id] = macro_mol_graphs[T_id]
#
#     gts = valid_triplets_y[:, -1]
#     for i in range(1, n_epoch + 1):
#         time_start = time.time()
#         if pred_mode == 'two_pred':
#             pred1, pred2 = model(curr_smalls, curr_biotechs, curr_targets,
#                                  nids['drug'], nids['target'],
#                                  sampled_inter_graph,
#                                  valid_triplets_y[:, :-1])
#             return gts, pred1, pred2
#         else:
#             pred = model(curr_smalls, curr_biotechs, curr_targets,
#                          nids['drug'], nids['target'],
#                          sampled_inter_graph,
#                          valid_triplets_y[:, :-1])
#         loss = torch.mean(loss_bce(pred, torch.tensor(gts).float().view(pred.shape)))
#         auc = metrics.roc_auc_score(gts, pred.detach().numpy())
#         opt.zero_grad()
#         loss.backward()
#         opt.step()
#         print(f"Epoch {i}: {loss.item()}  auc: {auc}  time: {time.time() - time_start}")


def train_BioMIP_on_sampled_subgraph(n_epoch,
                                     model,
                                     small_set, macro_set,
                                     small_mol_graphs, macro_mol_graphs,
                                     inter_graph,
                                     relation2id,
                                     triplet_ys,
                                     id2drug, id2target,
                                     opt,
                                     device,
                                     loss_bce,
                                     loss_kl,
                                     loss_wei=[0.1, 0.6],
                                     pred_mode='two_pred'
                                     ):
    etype_deg_dict = dict()

    sampled_inter_graph, nids, valid_triplets_y = sample_BioM_subgraph(inter_graph, relation2id, etype_deg_dict,
                                                                       [0.5, 0.5],
                                                                       triplet_ys)
    sampled_inter_graph = sampled_inter_graph.to(device)
    print("sampled_inter_graph: ", sampled_inter_graph)
    # else:
    #     raise NotImplementedError
    curr_smalls, curr_biotechs, curr_targets = {}, {}, {}
    for id in nids['drug']:
        DB_id = id2drug[id]
        if DB_id in small_set:
            curr_smalls[id] = small_mol_graphs[DB_id]
        else:  # is biotech drug
            curr_biotechs[id] = macro_mol_graphs[DB_id]

    for id in nids['target']:
        T_id = id2target[id]
        curr_targets[id] = macro_mol_graphs[T_id]

    gts = valid_triplets_y[:, -1]
    for i in range(1, n_epoch + 1):
        time_start = time.time()
        if pred_mode == 'two_pred':
            pred1, pred2 = model(curr_smalls, curr_biotechs, curr_targets,
                                 nids['drug'], nids['target'],
                                 sampled_inter_graph,
                                 valid_triplets_y[:, :-1])
            return gts, pred1, pred2
        else:
            pred = model(curr_smalls, curr_biotechs, curr_targets,
                         nids['drug'], nids['target'],
                         sampled_inter_graph,
                         valid_triplets_y[:, :-1])
        loss = torch.mean(loss_bce(pred, torch.tensor(gts).float().view(pred.shape).to(device)))
        auc = metrics.roc_auc_score(gts, pred.detach().cpu().numpy())
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"Epoch {i}: {loss.item()}  auc: {auc}  time: {time.time() - time_start}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    params = parser.parse_args()
    # initialize_experiment(params)
    # save featurized intra-view graphs
    if params.dataset == 'default_drugbank':
        params.aln_path = '/data/rzy/drugbank_prot/full_drugbank/aln'
        params.npy_path = '/data/rzy/drugbank_prot/full_drugbank/pconsc4'
        params.small_mol_db_path = f'/data/rzy/drugbank_prot/{params.dataset}/smile_graph_db_{params.SMILES_featurizer}'
        params.macro_mol_db_path = f'/data/rzy/drugbank_prot/{params.dataset}/prot_graph_db'  # _{params.prot_featurizer}
    elif params.dataset == 'davis':
        params.aln_path = '/data/rzy/davis/aln'
        params.npy_path = '/data/rzy/davispconsc4'
        params.small_mol_db_path = f'/data/rzy/davis/smile_graph_db_{params.SMILES_featurizer}'
        params.macro_mol_db_path = f'/data/rzy/davis/prot_graph_db'  # _{params.prot_featurizer}
    elif params.dataset == 'kiba':
        params.aln_path = '/data/rzy/kiba/aln'
        params.npy_path = '/data/rzy/kibapconsc4'
        params.small_mol_db_path = f'/data/rzy/kiba/smile_graph_db_{params.SMILES_featurizer}'
        params.macro_mol_db_path = f'/data/rzy/kiba/prot_graph_db'  # _{params.prot_featurizer}
    else:
        raise NotImplementedError

    # load the list of molecules with available chemical structures
    small_mol_list = pd.read_csv(f'data/{params.dataset}/SMILESstrings.csv', header=None, names=['id', '_'])[
        'id'].tolist()
    macro_mol_list = pd.read_csv(f'data/{params.dataset}/macro_seqs.csv', header=None, names=['id', '_'])['id'].tolist()
    # small_mol_list = [str(i) for i in small_mol_list]
    macro_mol_list = [str(i) for i in macro_mol_list]

    print('small molecule db_path:', params.small_mol_db_path)
    print('macro molecule db_path:', params.macro_mol_db_path)

    if not os.path.isdir(params.small_mol_db_path):
        generate_small_mol_graph_datasets(params)

    # if not processed, build intra-view graphs
    if not os.path.isdir(params.macro_mol_db_path):
        generate_macro_mol_graph_datasets(params)

    # load the inter-view graph
    adj_dict, triplets, \
    drug2id, target2id, relation2id, \
    id2drug, id2target, id2relation = build_inter_graph_from_links(
        params.dataset
    )

    inter_graph, relation2id = ssp_multigraph_to_dgl(
        adjs=adj_dict,
        relation2id=relation2id
    )

    Xy = convert_triplets_to_Xy(triplets['train'])

    # load intra-view graph dataset
    small_mol_graphs = IntraGraphDataset(db_path=params.small_mol_db_path, db_name='small_mol')
    macro_mol_graphs = IntraGraphDataset(db_path=params.macro_mol_db_path, db_name='macro_mol')

    params.atom_insize = small_mol_graphs.get_nfeat_dim()
    params.bond_insize = small_mol_graphs.get_efeat_dim()
    params.aa_node_insize = macro_mol_graphs.get_nfeat_dim()
    params.aa_edge_insize = macro_mol_graphs.get_efeat_dim()

    params.intra_out_dim = 200
    params.intra_small_layer_num = 2
    params.intra_macro_layer_num = 2
    params.intra_small_dropout = 0.
    params.intra_macro_dropout = 0.

    params.intra_out_dim = 200
    params.inter_emb_dim = 200
    params.attn_rel_emb_dim = 32
    params.num_rels = len(adj_dict)
    params.aug_num_rels = params.num_rels
    params.num_bases = 4  # 基数分解
    params.inter_layer_num = 2
    params.inter_dropout = 0.
    params.inter_edge_dropout = 0.
    params.inter_agg = 'sum'
    params.inter_has_attn = True

    params.pred_mode = 'two_pred'

    params.lr = 0.3
    # params.weight_decay
    if not params.disable_cuda and torch.cuda.is_available():
        params.device = torch.device('cuda:%d' % params.gpu)
    else:
        params.device = torch.device('cpu')

    # init model and loss
    params.rel2id = relation2id
    params.id2rel = {
        v: inter_graph.to_canonical_etype(k) for k, v in relation2id.items()
    }

    print(params)

    model = initialize_BioMIP(params)
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())

    print(f"total_num: {total_num}, trainable_num:{trainable_num}")

    loss_function_BCE = nn.BCEWithLogitsLoss(reduction='none')
    loss_function_KL = nn.KLDivLoss(reduction='none')

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=params.lr
                                 # ,weight_decay=args.weight_decay
                                 )

    train_BioMIP_on_sampled_subgraph(params.n_epoch, model, {str(i) for i in small_mol_list},
                                     {str(i) for i in macro_mol_list}, small_mol_graphs,
                                     macro_mol_graphs, inter_graph, relation2id, Xy, id2drug, id2target, optimizer,
                                     device=params.device, loss_bce=loss_function_BCE, loss_kl=loss_function_KL,
                                     pred_mode=params.mode)
