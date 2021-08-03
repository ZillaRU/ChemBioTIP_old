import os

import pandas as pd

all_dti = pd.read_csv('../full_drugbank/dti_pos.txt', sep='\t', header=None).values.tolist()
valid_drugs = set(list(pd.read_csv('../final_db/final_SMILESstrings.csv', header=None).iloc[:, 0])).union(
    set(list(pd.read_csv('../final_db/final_biodrugs.csv', header=None).iloc[:, 0]))
)
root_path = '/Users/zilla/PycharmProjects/drugbank_backup/preprocessDrugbank/preprocessDrugbank0512/csv_files'
#
macros = [s.replace('.npy', '') for s in os.listdir('../full_drugbank/macromolecules/pconsc4/')]
macro_set = set(macros)

valid_macro_drugs = set()
for i in macro_set:
    if i.startswith('DB'):
        valid_macro_drugs.add(i)
# valid_drugs = valid_small.union(valid_macro_drugs)
valid_targets = macro_set.difference(valid_macro_drugs)

res_list = []
tar_deg = {}
res_targets = set()
for i in all_dti:
    if i[2] in valid_drugs and i[0] in valid_targets:
        if i[0] in tar_deg:
            tar_deg[i[0]] += 1
        else:
            tar_deg[i[0]] = 1

for i in all_dti:
    if i[2] in valid_drugs and i[0] in tar_deg and tar_deg[i[0]] > 3:  # threshold3
        res_list.append(i)
        res_targets.add(i[0])

pd.DataFrame(list(res_targets)).to_csv('target---.csv', index=False, sep=',', header=None)

print(len(res_list))

pd.DataFrame(res_list).to_csv('dti---.csv', index=False, sep=',', header=None)

target_seq = pd.read_csv('../full_drugbank/macro_seqs.csv', header=None).values.tolist()

resres = []
for i in target_seq:
    if i[0] in res_targets:
        resres.append(i)
pd.DataFrame(resres).to_csv('target_seq.csv', index=False, sep=',', header=None)
