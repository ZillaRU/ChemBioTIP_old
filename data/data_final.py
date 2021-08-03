import os

import pandas as pd

# ddi_pd = pd.read_csv(
#     '/Users/zilla/PycharmProjects/drugbank_backup/preprocessDrugbank/preprocessDrugbank0512/drugbank/full_DDIs_mechanism_action.csv',
#     header=None)
# #
# root_path = '/Users/zilla/PycharmProjects/drugbank_backup/preprocessDrugbank/preprocessDrugbank0512/csv_files'
# #
# macros = [s.replace('.npy', '') for s in os.listdir('../data/full_drugbank/macromolecules/pconsc4/')]
# macro_set = set(macros)
# be_npy_map, npy_be_map = {}, {}
# npy_seq_map = {}
# for type in ['enzymes', 'transporters', 'targets', 'carriers']:
#     _df = pd.read_csv(f'{root_path}/{type}_polypeptides.csv')
#     be_npy_map = {**be_npy_map, **dict(zip(_df['parent_id'], _df['id']))}
#     npy_be_map = {**npy_be_map, **dict(zip(_df['id'], _df['parent_id']))}
#     npy_seq_map = {**npy_seq_map, **dict(zip(_df['id'], _df['amino_acid_sequence']))}
#
# print(len(be_npy_map))
# print(len(npy_be_map))
#
# macro_drug_db_seq = pd.read_csv(f'{root_path}/drug_sequences.csv')
# macro_drug_db_seq = dict(zip(macro_drug_db_seq['parent_key'], macro_drug_db_seq['sequence']))
#
# valid_npys = macro_set.intersection(be_npy_map.values()).intersection(npy_be_map.keys())
# valid_be_npy_map = {i: npy_be_map[i] for i in valid_npys}
# print("#valid target: ", len(valid_be_npy_map))
#
# drug_df = pd.read_csv(f'{root_path}/drug_calculated_properties.csv')
# valid_small = set(drug_df[drug_df['kind'] == 'SMILES']['parent_key'].values.tolist())
# valid_small_id_SMILES = drug_df[drug_df['kind'] == 'SMILES'][['parent_key', 'value']].values.tolist()
#
# valid_macro_drugs = set()
# for i in macro_set:
#     if i.startswith('DB'):
#         valid_macro_drugs.add(i)
# valid_drugs = valid_small.union(valid_macro_drugs)
# valid_targets = macro_set.difference(valid_macro_drugs)
#
# ddi_list = ddi_pd.values.tolist()
# res_list = []
# for act in ddi_list:
#     if id(act[0] in valid_drugs and act[1] in valid_drugs):
#         action = '+' if int(act[6]) == 1 else '-'
#         res_list.append([act[0], f'{act[5]}{action}', act[1]])
#
# pd.DataFrame(res_list).to_csv('final_db/ddi_valid_all.csv', index=False, sep=',', header=None)


dti_pd = pd.read_csv(
    '/Users/zilla/PycharmProjects/drugbank_backup/preprocessDrugbank/preprocessDrugbank0512/drugbank/full_DDIs_mechanism_action.csv',
    header=None)
#
root_path = '/Users/zilla/PycharmProjects/drugbank_backup/preprocessDrugbank/preprocessDrugbank0512/csv_files'
#
macros = [s.replace('.npy', '') for s in os.listdir('../data/full_drugbank/macromolecules/pconsc4/')]
macro_set = set(macros)
be_npy_map, npy_be_map = {}, {}
npy_seq_map = {}
for type in ['enzymes', 'transporters', 'targets', 'carriers']:
    _df = pd.read_csv(f'{root_path}/{type}_polypeptides.csv')
    be_npy_map = {**be_npy_map, **dict(zip(_df['parent_id'], _df['id']))}
    npy_be_map = {**npy_be_map, **dict(zip(_df['id'], _df['parent_id']))}
    npy_seq_map = {**npy_seq_map, **dict(zip(_df['id'], _df['amino_acid_sequence']))}

print(len(be_npy_map))
print(len(npy_be_map))

macro_drug_db_seq = pd.read_csv(f'{root_path}/drug_sequences.csv')
macro_drug_db_seq = dict(zip(macro_drug_db_seq['parent_key'], macro_drug_db_seq['sequence']))

valid_npys = macro_set.intersection(be_npy_map.values()).intersection(npy_be_map.keys())
valid_be_npy_map = {i: npy_be_map[i] for i in valid_npys}
print("#valid target: ", len(valid_be_npy_map))

drug_df = pd.read_csv(f'{root_path}/drug_calculated_properties.csv')
valid_small = set(drug_df[drug_df['kind'] == 'SMILES']['parent_key'].values.tolist())
valid_small_id_SMILES = drug_df[drug_df['kind'] == 'SMILES'][['parent_key', 'value']].values.tolist()

valid_macro_drugs = set()
for i in macro_set:
    if i.startswith('DB'):
        valid_macro_drugs.add(i)
valid_drugs = valid_small.union(valid_macro_drugs)
valid_targets = macro_set.difference(valid_macro_drugs)

dti_list = dti_pd.values.tolist()
res_list = []
for act in dti_list:
    if id(act[0] in valid_drugs and act[1] in valid_drugs):
        action = '+' if int(act[6]) == 1 else '-'
        res_list.append([act[0], f'{act[5]}{action}', act[1]])

pd.DataFrame(res_list).to_csv('final_db/dti_valid_all.csv', index=False, sep=',', header=None)
