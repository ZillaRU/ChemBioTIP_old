import torch.nn as nn
import dgl.function as fn

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
    def forward(self, graph, h, etype):
        with graph.local_scope():
            graph.ndata['h'] = h  # 一次性为所有节点类型的 'h'赋值
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']


class FinalPredictor(nn.Module):
    def forward(self, graph, h, etype):
        # h是从5.1节中对每种类型的边所计算的节点表示
        with graph.local_scope():
            graph.ndata['h'] = h  # 一次性为所有节点类型的 'h'赋值
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']
