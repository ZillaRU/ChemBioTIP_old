# import pickle
import dill


def serialize(data):
    data_tuple = tuple(data.values())
    return dill.dumps(data_tuple)


def deserialize_small(data):
    print('deserialize_small')
    keys = ('mol_id', 'mol_graph', 'graph_size')
    return dict(zip(keys, dill.loads(data)))


def deserialize_macro(data):
    print('deserialize_macro')
    keys = ('mol_id', 'seq', 'mol_graph')
    print(data)
    dill.loads(data)
    return dict(zip(keys, dill.loads(data)))