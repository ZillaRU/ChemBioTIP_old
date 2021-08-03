import pandas as pd

id_seq_list = pd.read_csv('/Users/zilla/PycharmProjects/BioMIP/data/full_drugbank/SMILESstrings.csv',
                          header=None).values.tolist()

invalid = set(pd.read_csv('/Users/zilla/PycharmProjects/BioMIP/data/full_drugbank/invalid_smiles.csv',
                          header=None).loc[:, 0].values.tolist())
# print(id_seq_list)
with open(f'/Users/zilla/PycharmProjects/BioMIP/data/full_drugbank/SMILESstrings1.csv', 'w') as f:
    for [id, seq] in id_seq_list:
        # print(id)
        if (id in invalid):
            continue
        temp = seq
        # temp = seq[seq.find("\n") + 1:].replace('\n', '')
        f.write(f"{id},{temp}\n")
        # {id},')
