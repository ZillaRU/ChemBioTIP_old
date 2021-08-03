import random

import dgl
import dgl.backend as F
import numpy as np
from dgl.dataloading.negative_sampler import _BaseNegativeSampler

from utils.graph_sampler import BioMIP_sampler


class NegativeSampler(object):
    def __init__(self, g, k):
        # 缓存概率分布
        self.weights = {
            etype: g.in_degrees(etype=etype).float() ** 0.75
            for _, etype, _ in g.canonical_etypes
        }
        self.k = k

    def __call__(self, g, eids_dict):
        result_dict = {}
        for etype, eids in eids_dict.items():
            src, _ = g.find_edges(eids, etype=etype)
            src = src.repeat_interleave(self.k)
            dst = self.weights[etype].multinomial(len(src), replacement=True)
            result_dict[etype] = (src, dst)
        return result_dict


class TypeRelUniform(_BaseNegativeSampler):
    """Negative sampler that randomly chooses negative destination nodes
    for each source node according to a uniform distribution.

    For each edge ``(u, v)`` of type ``(srctype, etype, dsttype)``, DGL generates
    :attr:`k` pairs of negative edges ``(u, v')``, where ``v'`` is chosen
    uniformly from all the nodes of type ``dsttype``.  The resulting edges will
    also have type ``(srctype, etype, dsttype)``.

    Parameters
    ----------
    k : int
        The number of negative examples per edge.
    """

    def __init__(self, k):
        self.k = k

    def _generate(self, g, eids, canonical_etype):
        _, _, vtype = canonical_etype
        shape = F.shape(eids)
        dtype = F.dtype(eids)
        ctx = F.context(eids)
        shape = (shape[0] * self.k,)
        src, _ = g.find_edges(eids, etype=canonical_etype)
        src = F.repeat(src, self.k, 0)
        dst = F.randint(shape, dtype, ctx, 0, g.number_of_nodes(vtype))
        return src, dst


def sample_BioM_subgraph(full_graph, relation2id,
                         etype_deg_dict,
                         sample_rate: list,
                         triplet_ys) -> (dgl.DGLGraph, dict, list):
    n_drug, n_target = full_graph.num_nodes('drug'), full_graph.num_nodes('target')
    drug_list, target_list = list(range(n_drug)), list(range(n_target))
    # todo: modify here
    sample_drug_idx = random.sample(drug_list, int(sample_rate[0] * n_drug))
    sample_target_idx = random.sample(target_list, int(sample_rate[1] * n_target))

    # return dgl.node_subgraph(full_graph,{})
    triplet_ys = triplet_ys.tolist()
    drug_idx_set, target_idx_set = set(sample_drug_idx), set(sample_target_idx)
    meta_rel_dict = dict(zip(full_graph.etypes, full_graph.canonical_etypes))
    # print(meta_rel_dict)
    rel_id_dict = {
        relation2id[k]: v for k, v in meta_rel_dict.items() if not k[0].startswith('~')
    }
    valid_Xys = []
    for [u, v, r, l] in triplet_ys:
        u_drug = rel_id_dict[r][0] == 'drug'
        v_drug = rel_id_dict[r][2] == 'drug'
        u_valid = u in drug_idx_set if u_drug else u in target_idx_set
        if u_valid:
            v_valid = v in drug_idx_set if v_drug else v in target_idx_set
        else:
            continue
        if v_valid:
            valid_Xys.append([u, v, r, l])

    print(len(sample_drug_idx), len(sample_target_idx))
    sample_idx = {
        'drug': sample_drug_idx,
        'target': sample_target_idx
    }
    return dgl.node_subgraph(full_graph, sample_idx), sample_idx, np.array(valid_Xys)


def get_bioM_minibatch_dataloader(inter_graph):
    def get_sampled_drugs_and_targets():
        pass

    # node_list = list(range(n_node))
    # sample_idx = random.sample(node_list, sample_size)

    train_eid_dict = {
        etype: inter_graph.edges(etype=etype, form='eid')
        for etype in inter_graph.etypes
    }

    # sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    sampler = BioMIP_sampler(2)
    dataloader = dgl.dataloading.EdgeDataLoader(
        inter_graph,
        train_eid_dict,
        sampler
    )
    return dataloader
