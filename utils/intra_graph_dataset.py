import lmdb
from torch.utils.data import Dataset

from utils.data_utils import deserialize_small, deserialize_macro


class IntraGraphDataset(Dataset):
    def __init__(self, db_path, db_name):
        self.main_env = lmdb.open(db_path, readonly=True, max_dbs=3, lock=False)
        self.db = self.main_env.open_db(db_name.encode())
        self.deserialize = deserialize_small if db_name == 'small_mol' else deserialize_macro
        self.__getitem__(0)  # test

    def __getitem__(self, index):
        with self.main_env.begin(db=self.db) as txn:
            str_id = '{:08}'.format(index).encode('ascii')
            mol_id, graph_or_seq, size_or_graph = self.deserialize(txn.get(str_id)).values()
            print(mol_id, graph_or_seq, size_or_graph)
        return mol_id, graph_or_seq, size_or_graph

