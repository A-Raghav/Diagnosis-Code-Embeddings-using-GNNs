import pandas as pd
import numpy as np

hcc_weights = pd.read_csv("graphs-eda/data/metadata/hcccoefn.csv")
hcc_weights = hcc_weights.T.reset_index()
hcc_weights.columns = ['hcc','weight']
hcc_filter = hcc_weights.hcc.str.contains(r"CE_HCC[\d]+")
hcc_weights = hcc_weights[hcc_filter].reset_index(drop=True)
hcc_weights.hcc = hcc_weights.hcc.str.findall(r"[\d]+").apply(lambda x: int(x[0]))
hcc_weights['cat'] = np.nan

first_flag = True
cat = 1
for index, row in hcc_weights.iterrows():
    if first_flag==True:
        first_flag=False
        hcc = row['hcc']
        weight = row['weight']
        hcc_weights.loc[index, 'cat'] = f"CAT-{cat}"
        continue
    if row['weight']>weight:
        cat+=1
    hcc_weights.loc[index, 'cat'] = f"CAT-{cat}"
    hcc = row['hcc']
    weight = row['weight']

hcc_weights.to_csv("hcc_weights_categorised.csv", index=False)