import dgl


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


def get_dgl_minibatch_dataloader(inter_graph):
    """
    Args:
        inter_graph: training inter-graph

    Returns:

    """
    train_eid_dict = {
        etype: inter_graph.edges(etype=etype, form='eid')
        for etype in inter_graph.etypes
    }
    print(train_eid_dict)
    print(inter_graph.etypes)
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    dataloader = dgl.dataloading.EdgeDataLoader(
        inter_graph,
        train_eid_dict,
        sampler,
        negative_sampler=dgl.dataloading.negative_sampler.Uniform(5),
        # negative_sampler=NegativeSampler(inter_graph, 5)
        # batch_size=1024,
        # shuffle=True,
        # drop_last=False,
        # num_workers=4
    )
    return dataloader
