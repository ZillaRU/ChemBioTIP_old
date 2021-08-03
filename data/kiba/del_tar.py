import pandas as pd

dels = {'P08922', 'P42345', 'P78527', 'Q5S007'}
files = [
    'train_pos',
    'train_neg',
    'valid_pos',
    'valid_neg',
    'test_pos',
    'test_neg'
]
ds = "davis"
res = []
for f in files:
    for [d, t] in pd.read_csv(f'data/{ds}/{f}.csv', header=None).values.tolist():
        if t not in dels:
            res.append([d, "dt", t])
    pd.DataFrame(res).to_csv(f'data/{ds}/{f}.csv', header=None, index=False)
