import torch
import torch.nn as nn

# 分 batch

'''
https://docs.dgl.ai/generated/dgl.DGLGraph.apply_edges.html?highlight=update

func (dgl.function.BuiltinFunction or callable) – 
The function to generate new edge features. 
It must be either a DGL Built-in Function or a User-defined Functions.

edges (edges) –
The edges to update features on. The allowed input formats are:
int: A single edge ID.
Int Tensor: Each element is an edge ID. The tensor must have the same device type and ID data type as the graph’s.
iterable[int]: Each element is an edge ID.
(Tensor, Tensor): The node-tensors format where the i-th elements of the two tensors specify an edge.
(iterable[int], iterable[int]): Similar to the node-tensors format but stores edge endpoints in python iterables.
Default value specifies all the edges in the graph.

etype (str or (str, str, str), optional) –
The type name of the edges. The allowed type name formats are:
(str, str, str) for source node type, edge type and destination node type.
or one str edge type name if the name can uniquely identify a triplet format in the graph.

Can be omitted if the graph has only one type of edges.
'''


class IntraPredictor(nn.Module):
    def __init__(self,
                 num_rels,
                 rel_emb_dim):
        super().__init__()
        self.rel_emb_intra = nn.Embedding(num_rels, rel_emb_dim, sparse=False)

    def u_op_v(self, edges):
        # print(edges)
        return {
            'intra_score': torch.cat(
                [
                    edges.src['intra'],
                    edges.dst['intra'],
                    self.rel_emb_intra[edges.data['type']]
                ],
                dim=1
            )
        }

    def forward(self, graph, triplets):
        pass

    # with graph.local_scope():
    #     graph.apply_edges(self.u_op_v, etype=etype)
    #     return graph.edges[etype].data['score']


class FinalPredictor(nn.Module):
    def __init__(self,
                 num_rels,
                 rel_emb_dim):
        super(FinalPredictor, self).__init__()
        self.rel_emb_inter = nn.Embedding(num_rels, rel_emb_dim, sparse=False)

    def u_op_v(self, edges):
        # print(edges)
        return {
            'score': torch.cat(
                [
                    edges.src['intra'],
                    edges.dst['intra'],
                    edges.src['repr'],
                    edges.dst['repr'],
                    self.rel_emb_inter[edges.data['type']]
                ],
                dim=1
            )
        }

    def forward(self, graph, triplets):
        pass
        # with graph.local_scope():
        #     graph.apply_edges(self.u_op_v)
        # return graph.edges.data['score']


# class HardMixPredictor(nn.Module):
#     def __init__(self,
#                  num_rels,
#                  rel_emb_dim,
#                  intra_fc_in,
#                  inter_fc_in):
#         super(HardMixPredictor, self).__init__()
#         self.rel_emb_intra = nn.Embedding(num_rels, rel_emb_dim, sparse=False)
#         self.rel_emb_inter = nn.Embedding(num_rels, rel_emb_dim, sparse=False)
#         self.fc_intra = nn.Linear(in_features=2*intra_fc_in+rel_emb_dim, out_features=1)
#         self.fc_inter = nn.Linear(in_features=inter_fc_in+rel_emb_dim, out_features=1)
#         # self.fc_inter = nn.Linear(in_features=2*inter_fc_in+rel_emb_dim, out_features=1)
#
#     def u_op_v(self, edges):
#         shape0 = list(edges.src['repr'].shape)[0]
#         return {
#             'score_intra': torch.cat(
#                 (
#                     edges.src['intra'],
#                     edges.dst['intra'],
#                     self.rel_emb_intra(edges.data['type'])
#                 ),
#                 dim=1
#             ),
#             'score_inter': torch.cat(
#                 (
#                     edges.src['repr'].view(shape0, -1),
#                     edges.dst['repr'].view(shape0, -1),
#                     self.rel_emb_inter(edges.data['type'])
#                 ),
#                 dim=1
#             )
#         }
#
#     def forward(self, graph, a1=0.5, a2=0.5):
#         # with graph.local_scope():
#         #     for _, etype, _ in graph.canonical_etypes:
#         #         # graph.apply_edges(self.u_op_v, etype=etype)
#         #         print(etype)
#         # print(graph)
#         # return a1 * self.fc_intra(graph.edges[etype]['score_intra']) + \
#         #        a2 * self.fc_inter(graph.edges[etype]['score_inter'])
#         with graph.local_scope():
#             for _,rel,_ in graph.canonical_etypes:
#                 graph.apply_edges(self.u_op_v, etype=rel)
#             return a1 * self.fc_intra(graph.edata['score_intra']) + \
#                 a2 * self.fc_inter(graph.edata['score_inter'])

# class LastPredictor(nn.Module):
#     def __init__(self):
#         super(LastPredictor, self).__init__()
#     def forward(self, graph, samples):
#         graph.nodes[].data[] =


class SoftMixPredictor(nn.Module):
    def __init__(self):
        super(SoftMixPredictor, self).__init__()

    def forward(self):
        pass


class HardMixPredictor(nn.Module):
    def __init__(self,
                 num_rels,
                 id2rel,
                 rel_emb_dim,
                 device,
                 intra_fc_in,
                 inter_fc_in):
        super(HardMixPredictor, self).__init__()
        self.device = device
        self.id2rel = id2rel
        self.in1 = 2 * intra_fc_in + rel_emb_dim
        self.in2 = 2 * inter_fc_in + rel_emb_dim
        self.rel_emb_intra = nn.Embedding(num_rels, rel_emb_dim, sparse=False).to(device)
        self.rel_emb_inter = nn.Embedding(num_rels, rel_emb_dim, sparse=False).to(device)
        self.fc_intra = nn.Linear(in_features=self.in1, out_features=1).to(device)
        self.fc_inter = nn.Linear(in_features=self.in2, out_features=1).to(device)
        self.act = nn.Tanh().to(device)  # todo Sigmoid ReLU Tanh

    def forward(self,
                d_intra, d_inter, did_sub,
                t_intra, t_inter, tid_sub,
                triplets, a1=0.6, a2=0.2):
        intra_tensor = torch.zeros((triplets.shape[0], self.in1)).to(self.device)
        inter_tensor = torch.zeros((triplets.shape[0], self.in2)).to(self.device)
        i = 0
        for tri in triplets:
            # print(tri)
            u, v, r = tri
            utype, _, vtype = self.id2rel[r]
            r = torch.tensor(r).long().to(self.device)
            intra_tensor[i] = torch.cat((d_intra[did_sub[u]] if utype == 'drug' else t_intra[tid_sub[u]],
                                         d_intra[did_sub[v]] if vtype == 'drug' else t_intra[tid_sub[v]],
                                         self.rel_emb_intra(r))).to(self.device)
            inter_tensor[i] = torch.cat((d_inter[did_sub[u]] if utype == 'drug' else t_inter[tid_sub[u]],
                                         d_inter[did_sub[v]] if vtype == 'drug' else t_inter[tid_sub[v]],
                                         self.rel_emb_inter(r))).to(self.device)
        return a1 * self.act(self.fc_intra(intra_tensor)) + a2 * self.act(self.fc_inter(inter_tensor))
        #
        # # triplets = triplets.T
        #
        # print(type(triplets))
