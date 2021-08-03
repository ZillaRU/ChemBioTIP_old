import pandas as pd

# for full drugbank

small_id_set = pd.read_csv('/Users/zilla/PycharmProjects/BioMIP/data/full_drugbank/SMILESstrings.csv', header=None,
                           usecols=[0]).values.tolist()
small_id_set = set([i[0] for i in small_id_set])
macro_id_set = pd.read_csv('/Users/zilla/PycharmProjects/BioMIP/data/full_drugbank/macro_seqs.csv', header=None,
                           usecols=[0]).values.tolist()
macro_id_set = set([i[0] for i in macro_id_set])
# biotech_id_set = pd.read_csv('/Users/zilla/PycharmProjects/BioMIP/data/full_drugbank//biodrugs.csv', header=None,
#                              usecols=[0]).values.tolist()
# biotech_id_set = set([i[0] for i in biotech_id_set])

# valid_drugs = small_id_set.union(biotech_id_set)

list_ddi = pd.read_csv('ddi_pos.txt', header=None, sep='\t').values.tolist()
print("df_ddi: ", len(list_ddi))
# valid_ddis = []
# for i in list_ddi:
#     if i[0] in valid_drugs and i[2] in valid_drugs:
#         valid_ddis.append(i)
# pd.DataFrame(valid_ddis).to_csv('valid_ddi_pos.txt', index=None, header=None)
#
# list_dti = pd.read_csv('dti_pos.txt', header=None, sep='\t').values.tolist()
# print("df_dti: ", len(list_dti))
# valid_dtis = []
# for i in list_dti:
#     if i[0] in macro_id_set and i[2] in valid_drugs:
#         valid_dtis.append(i)
#
# print(len(valid_ddis), len(valid_dtis))
#
# pd.DataFrame(valid_dtis).to_csv('valid_dti_pos.txt', index=None, header=None)

# df_ddi:  1138819
# df_dti:  25872
# 1136407 13922

valid_ddis = []
for i in list_ddi:
    if i[0] in small_id_set and i[2] in small_id_set:
        valid_ddis.append(i)
pd.DataFrame(valid_ddis).to_csv('../small_target_drugbank/valid_ddi_pos.txt', index=None, header=None)

list_dti = pd.read_csv('dti_pos.txt', header=None, sep='\t').values.tolist()
print("df_dti: ", len(list_dti))
valid_dtis = []
for i in list_dti:
    if i[0] in macro_id_set and i[2] in small_id_set:
        valid_dtis.append(i)

print(len(valid_ddis), len(valid_dtis))

pd.DataFrame(valid_dtis).to_csv('../small_target_drugbank/valid_dti_pos.txt', index=None, header=None)

# df_ddi:  1138819
# df_dti:  25872
# 1114713 13922
