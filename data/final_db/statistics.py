import pandas as pd

# valid small
id_smiles = pd.read_csv('../full_drugbank/SMILESstrings.csv').values.tolist()
id_seqs = pd.read_csv('../full_drugbank/biodrugs.csv').values.tolist()

valid_small_bio = set(list(pd.read_csv('drugs_valid_th30000_10.csv').iloc[:, 0]))

res_small = []
smalls = set()
for i in id_smiles:
    if i[0] in valid_small_bio:
        res_small.append(i)
        smalls.add(i[0])

res_bio = []
bios = set()
for i in id_seqs:
    if i[0] in valid_small_bio:
        res_bio.append(i)
        bios.add(i[0])

print(len(res_small), len(res_bio))
pd.DataFrame(res_small).to_csv('final_SMILESstrings.csv', header=None, index=False)
pd.DataFrame(res_bio).to_csv('final_biodrugs.csv', header=None, index=False)

print(valid_small_bio.difference(set(bios)).difference(set(smalls)))

ss, sm, mm, _del = 0, 0, 0, 0
res = []
ddis = pd.read_csv('ddi_valid_th30000_10_directed.csv').values.tolist()
for i in ddis:
    if i[0] in smalls and i[2] in smalls:
        ss += 1
        res.append(i)
    elif (i[0] in smalls and i[2] in bios) or (i[2] in smalls and i[0] in bios):
        sm += 1
        res.append(i)
    elif i[0] in bios and i[2] in bios:
        mm += 1
        res.append(i)
    else:
        _del += 1
print(ss, sm, mm, _del)
pd.DataFrame(res).to_csv('ddi---.csv', index=False, sep=',', header=None)

# valid biotech

# avg smiles len
# avg amino acid seq len

# avg small molecule nodes
# avg small molecule edges

# avg macro nodes
# avg macro edges
