import logging
import multiprocessing as mp
import os
import struct
from functools import partial

import dgl
import lmdb
import numpy as np
import pandas as pd
import rdkit.Chem as chem
import torch
from dgllife.utils import mol_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer, BaseAtomFeaturizer, \
    ConcatFeaturizer, atom_type_one_hot, atom_degree_one_hot, atom_formal_charge, atom_num_radical_electrons, \
    atom_hybridization_one_hot, atom_total_num_H_one_hot, BaseBondFeaturizer, one_hot_encoding, \
    AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer, PAGTNAtomFeaturizer, PAGTNEdgeFeaturizer
from tqdm import tqdm

from utils.data_utils import serialize
from utils.mol_utils import PSSM_calculation, pro_res_table, pro_res_aliphatic_table, pro_res_polar_neutral_table, \
    pro_res_acidic_charged_table, pro_res_basic_charged_table, pro_res_aromatic_table, res_pkb_table, res_pkx_table, \
    res_hydrophobic_ph7_table, res_pka_table, res_hydrophobic_ph2_table, res_weight_table, res_pl_table, \
    one_of_k_encoding


def build_molecule_graph(args_, graph_mode='bigraph', featurizer='base'):
    """Construct graphs from SMILES and featurize them
    options:
    BaseAtomFeaturizer
    CanonicalAtomFeaturizer
    PretrainAtomFeaturizer
    WeaveAtomFeaturizer
    AttentiveFPAtomFeaturizer

    CanonicalBondFeaturizer
    WeaveEdgeFeaturizer
    PretrainBondFeaturizer
    AttentiveFPBondFeaturizer
    PAGTNEdgeFeaturizer

    Returns
    -------
    DGLGraph
        a graph constructed and featurized or None
        parsed by RDKit
    """
    idx, (mol_id, smiles) = args_
    mol = chem.MolFromSmiles(smiles)
    if graph_mode == 'bigraph' and featurizer == 'base':
        g = mol_to_bigraph(mol,
                           add_self_loop=True,
                           node_featurizer=PretrainAtomFeaturizer(),
                           edge_featurizer=PretrainBondFeaturizer(),
                           canonical_atom_order=False)
    elif graph_mode == 'bigraph' and featurizer == 'afp':  # Attentive FP
        g = mol_to_bigraph(mol,
                           add_self_loop=True,
                           node_featurizer=AttentiveFPAtomFeaturizer(atom_data_field='hv'),
                           edge_featurizer=AttentiveFPBondFeaturizer(bond_data_field='he')
                           )
    elif graph_mode == 'bigraph' and featurizer == 'pagtn':  # PAGTN
        g = mol_to_bigraph(mol,
                           add_self_loop=True,
                           node_featurizer=PAGTNAtomFeaturizer(atom_data_field='hv'),
                           edge_featurizer=PAGTNEdgeFeaturizer(bond_data_field='he')
                           )
    else:  # custom
        def chirality(atom):
            try:
                return one_hot_encoding(atom.GetProp('_CIPCode'), ['R', 'S']) + \
                       [atom.HasProp('_ChiralityPossible')]
            except:
                return [False, False] + [atom.HasProp('_ChiralityPossible')]

        atom_featurizer = BaseAtomFeaturizer(
            {'hv': ConcatFeaturizer([
                partial(atom_type_one_hot, allowable_set=[
                    'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At'],
                        encode_unknown=True),
                partial(atom_degree_one_hot, allowable_set=list(range(6))),
                atom_formal_charge, atom_num_radical_electrons,
                partial(atom_hybridization_one_hot, encode_unknown=True),
                lambda atom: [0],  # A placeholder for aromatic information,
                atom_total_num_H_one_hot, chirality
            ],
            )})
        bond_featurizer = BaseBondFeaturizer({
            'he': lambda bond: [0 for _ in range(10)]
        })
        g = mol_to_bigraph(mol,
                           add_self_loop=True,
                           node_featurizer=atom_featurizer,
                           edge_featurizer=bond_featurizer
                           )
    datum = {
        'mol_id': mol_id,
        'mol_graph': g,
        'graph_size': g.num_nodes()
    }
    idx = '{:08}'.format(idx).encode('ascii')
    return (idx, datum)


def init_folder(params, _):
    global _dataset
    _dataset = params


def residue_features(residue):
    res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 1 if residue in pro_res_aromatic_table else 0,
                     1 if residue in pro_res_polar_neutral_table else 0,
                     1 if residue in pro_res_acidic_charged_table else 0,
                     1 if residue in pro_res_basic_charged_table else 0]
    res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue], res_pkx_table[residue],
                     res_pl_table[residue], res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]
    # print(np.array(res_property1 + res_property2).shape)
    return np.array(res_property1 + res_property2)


def seq_feature(pro_seq):
    pro_hot = np.zeros((len(pro_seq), len(pro_res_table)))
    pro_property = np.zeros((len(pro_seq), 12))
    for i in range(len(pro_seq)):
        pro_hot[i,] = one_of_k_encoding(pro_seq[i], pro_res_table)
        pro_property[i,] = residue_features(pro_seq[i])
    return np.concatenate((pro_hot, pro_property), axis=1)


def macro_mol_feature(aln, seq):
    pssm = PSSM_calculation(aln, seq)
    other_feature = seq_feature(seq)
    return np.concatenate((np.transpose(pssm, (1, 0)), other_feature), axis=1)


def macro_mol_to_feature(seq, aln):
    feature = macro_mol_feature(aln, seq)
    return feature


def build_seq_to_graph(args):
    # print(os.getcwd())
    idx, (mol_id, seq) = args
    aln_path = f'./data/{_dataset}/macromolecules/aln/{mol_id}.aln'
    cmap_path = f'./data/{_dataset}/macromolecules/pconsc4/{mol_id}.npy'
    macro_mol_edge_index = []
    contact_map = np.load(cmap_path, allow_pickle=True)
    contact_map += np.matrix(np.eye(contact_map.shape[0]))
    index_row, index_col = np.where(contact_map >= 0.5)
    for i, j in zip(index_row, index_col):
        macro_mol_edge_index.append([i, j])
    mol_feature = torch.from_numpy(macro_mol_to_feature(seq, aln_path))
    g = dgl.graph((torch.from_numpy(index_row), torch.from_numpy(index_col)))
    # g = dgl.heterograph((index_row, index_col))
    g.ndata['hv'] = mol_feature
    # g.edata['he'] =
    datum = {
        'mol_id': mol_id,
        'seq': seq,
        'mol_graph': g
    }
    idx = '{:08}'.format(idx).encode('ascii')
    print(idx, datum)
    return (idx, datum)


def generate_small_mol_graph_datasets(params):
    logging.info(f"Construct intra-view graphs for small molecules in {params.dataset}...")

    SMILES_csv = pd.read_csv(f'data/{params.dataset}/SMILESstrings.csv')

    # todo: fix map_size
    env = lmdb.open(params.small_mol_db_path, map_size=1e9, max_dbs=6)

    num_mol = SMILES_csv.shape[0]

    with env.begin(write=True, db=env.open_db('small_mol'.encode())) as txn:
        txn.put('num_graphs'.encode(), num_mol.to_bytes(int.bit_length(num_mol), byteorder='little'))

    graph_sizes = []
    with mp.Pool(processes=None) as p:
        args_ = zip(range(num_mol), np.array(SMILES_csv).tolist())
        for (idx, datum) in tqdm(p.imap(build_molecule_graph, args_), total=num_mol):
            graph_sizes.append(datum['graph_size'])
            with env.begin(write=True, db=env.open_db('small_mol'.encode())) as txn:
                txn.put(idx, serialize(datum))

    with env.begin(write=True) as txn:
        print('==== ==== ==== ==== ==== ==== ==== Writing ==== ==== ==== ==== ==== ==== ====')
        txn.put('avg_molgraph_size'.encode(), struct.pack('f', float(np.mean(graph_sizes))))
        txn.put('min_molgraph_size'.encode(), struct.pack('f', float(np.min(graph_sizes))))
        txn.put('max_molgraph_size'.encode(), struct.pack('f', float(np.max(graph_sizes))))
        txn.put('std_molgraph_size'.encode(), struct.pack('f', float(np.std(graph_sizes))))


def generate_macro_mol_graph_datasets(params):
    logging.info(f"Construct intra-view graphs for macro molecules in {params.dataset}...")

    seq_csv = pd.read_csv(f'data/{params.dataset}/macro_seqs.csv')

    # todo: fix map_size
    env = lmdb.open(params.macro_mol_db_path, map_size=1e9, max_dbs=6)

    num_mol = seq_csv.shape[0]

    with env.begin(write=True, db=env.open_db('macro_mol'.encode())) as txn:
        txn.put('num_graphs'.encode(), num_mol.to_bytes(int.bit_length(num_mol), byteorder='little'))

    graph_sizes = []

    with mp.Pool(processes=None, initializer=init_folder, initargs=(params.dataset, 0)) as p:
        args_ = zip(range(num_mol), np.array(seq_csv).tolist())
        for (idx, datum) in tqdm(p.imap(build_seq_to_graph, args_), total=num_mol):
            graph_sizes.append(len(datum['seq']))
            with env.begin(write=True, db=env.open_db('small_mol'.encode())) as txn:
                txn.put(idx, serialize(datum))

    with env.begin(write=True) as txn:
        print('==== ==== ==== ==== ==== ==== ==== Writing ==== ==== ==== ==== ==== ==== ====')
        txn.put('avg_molgraph_size'.encode(), struct.pack('f', float(np.mean(graph_sizes))))
        txn.put('min_molgraph_size'.encode(), struct.pack('f', float(np.min(graph_sizes))))
        txn.put('max_molgraph_size'.encode(), struct.pack('f', float(np.max(graph_sizes))))
        txn.put('std_molgraph_size'.encode(), struct.pack('f', float(np.std(graph_sizes))))
