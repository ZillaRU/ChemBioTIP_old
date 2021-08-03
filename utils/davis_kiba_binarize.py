from tdc.multi_pred import DTI

data = DTI(name='KIBA')

# data.convert_to_log(form='standard')  # data.convert_to_log(form='binding')
# convert back: data.convert_from_log(form = 'standard') # convert back: data.convert_from_log(form = 'binding')
split = data.get_split()
print(split)
data.binarize(threshold=11.5, order='descending')

data.label_distribution()
# split = data.get_split()
# print(split)
split = data.get_split()
for _set in ['train', 'valid', 'test']:
    _df = split[_set]
    _df.loc[_df['Y'] == 1, ['Drug_ID', 'Target_ID']].to_csv(f'../data/kiba/{_set}_pos.csv', index=None)
    _df.loc[_df['Y'] == 0, ['Drug_ID', 'Target_ID']].to_csv(f'../data/kiba/{_set}_neg.csv', index=None)
