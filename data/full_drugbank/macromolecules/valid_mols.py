import os

import pandas as pd

root_path = '/Users/zilla/PycharmProjects/drugbank_backup/preprocessDrugbank/preprocessDrugbank0512/csv_files'
#
macros = [s.replace('.npy', '') for s in os.listdir('pconsc4/')]
macro = set(macros)
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

valid_npys = macro.intersection(be_npy_map.values()).intersection(npy_be_map.keys())
# valid_be_npy_map = {i: npy_be_map[i] for i in valid_npys}
# print("#valid target: ", len(valid_be_npy_map))

# drug_df = pd.read_csv(f'{root_path}/drug_calculated_properties.csv')

# valid_small = set(drug_df[drug_df['kind'] == 'SMILES']['parent_key'].values.tolist())

# valid_small_id_SMILES = drug_df[drug_df['kind'] == 'SMILES'][['parent_key','value']].values.tolist()

# with open(f'../SMILESstrings.csv', 'w') as f:
#     for [i,j] in valid_small_id_SMILES:
#         f.write(f'{i},{j}\n')

valid_macro = set()
# print(valid_small)
# print(len(valid_small))
for i in macro:
    if i.startswith('DB'):
        valid_macro.add(i)
        print(i)
# # print(valid_macro)
# print(len(valid_macro))
# valid_drugs = valid_small.union(valid_macro)
#
# print("#valid drug: ", len(valid_drugs))
#
valid_prot_drug_and_target = valid_macro.union(valid_npys)
with open(f'../macro_seqs.csv', 'w') as f:
    for i in valid_npys:
        f.write(f'{i},"{npy_seq_map[i]}"\n')
    for k, v in macro_drug_db_seq.items():
        f.write(f'{k},"{v}"\n')

# for type in ['enzymes', 'transporters', 'targets', 'carriers']:
#     _df = pd.read_csv(f'{root_path}/{type}.csv')
#     be_db_list = _df[['id','parent_key']].values.tolist()
#     with open(f'../dti_pos_{type}.txt','w') as f:
#         for [be, db] in be_db_list:
#             if be in be_npy_map.keys() and db in valid_drugs:
#                 f.write(f'{be_npy_map[be]}\t{type}\t{db}\n')

# cnt = 0
# ddi_list = pd.read_csv(
#     '/Users/zilla/PycharmProjects/preprocessDrugbank/csv_files/final_u_r_v/ddi.csv').values.tolist()  # [['A_id', 'B_id']]
# with open(f'../ddi_pos.txt', 'w') as f:
#     for [drug1, rel, drug2] in ddi_list:
#         if drug1 < drug2 and drug1 in valid_drugs and drug2 in valid_drugs:
#             f.write(f'{drug1}\t{rel}\t{drug2}\n')
#             cnt += 1
# print(len(ddi_list), cnt)
# 2682156 1138819
# 2682156 2277637
