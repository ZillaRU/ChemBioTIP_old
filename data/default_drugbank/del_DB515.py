import pandas as pd

res = []
ddis = pd.read_csv('ddi_pos.csv', header=None).values.tolist()
for i in ddis:
    if i[0] == 'DB00515' or i[2] == 'DB00515':
        continue
    else:
        res.append(i)
pd.DataFrame(res).to_csv('ddi__.csv', header=None, index=False)
