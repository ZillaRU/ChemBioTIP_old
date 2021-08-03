import dgl
from dgl import utils, batch


def gen_neg_graph(pos_graph, intra_feats, rel_rmb) -> dgl.DGLGraph:
    """
    generate a negative graph with the nodes in pos_graph,
    the edges in negative graph
    Args:
        pos_graph
        intra_feats

    Returns:

    References:
        https://docs.dgl.ai/en/latest/generated/dgl.add_reverse_edges.html#dgl.add_reverse_edges
    """
    node_frames = utils.extract_node_subframes(pos_graph, None)
    neg_graph = dgl.DGLGraph()
    utils.set_new_frames(neg_graph, node_frames=node_frames)
    # for each edge (u, r, v),
    #   ~(ignore missing problem) if intra feat of u or v is missing~
    #   sample
    return neg_graph
