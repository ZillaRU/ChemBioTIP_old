import pandas as pd

ddi_list = pd.read_csv('ddi_valid_all.csv', header=None).values.tolist()
type_cnt = dict()
for i in ddi_list:
    if i[1] in type_cnt:
        type_cnt[i[1]] += 1
    else:
        type_cnt[i[1]] = 1

print(len(type_cnt))

major_ddi_cnt = dict()
for k, v in type_cnt.items():
    if v >= 30000:  # threshold1
        major_ddi_cnt[k] = v

print(major_ddi_cnt)

res_list = []
deg_dict = {}
for i in ddi_list:
    if i[1] in major_ddi_cnt and i[0] < i[2]:
        if i[0] in deg_dict:
            deg_dict[i[0]] += 1
        else:
            deg_dict[i[0]] = 1
        if i[2] in deg_dict:
            deg_dict[i[2]] += 1
        else:
            deg_dict[i[2]] = 1
        res_list.append(i)

print(len(res_list))

# hist = dict()
# for k, v in deg_dict.items():
#     if v in hist:
#         hist[v] += 1
#     else:
#         hist[v] = 1
valid_drugs = set()
for k, v in deg_dict.items():
    if v >= 10:  # threshold2
        valid_drugs.add(k)

pd.DataFrame(list(valid_drugs)).to_csv('drugs_valid_th30000_10.csv', index=False, header=None)

res_list_final = []
for i in res_list:
    if i[0] in valid_drugs and i[2] in valid_drugs:
        res_list_final.append(i)

print(len(res_list_final))

pd.DataFrame(res_list_final).to_csv('ddi_valid_th30000_10_directed.csv', index=False, sep=',', header=None)

# valid_drugs_2 =
