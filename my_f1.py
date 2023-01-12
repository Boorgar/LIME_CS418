#%%
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd


l = [[440, 22551, 2944, 4065],
[1, 1292, 22913, 1895, 3900],
[2, 1681, 20276, 2360, 2861],
[3, 339 , 24141, 3513, 2275]]

df = pd.DataFrame(l, columns=['', '0', '1', '2', '3'])

sn.heatmap(df, annot=True, fmt='g', cmap='Blues')

plt.show()
# %%
import json
with open('D:\Documents\GitHub\LIME_CS418\pseudo_agnews.json') as rf:
    data = json.load(rf)
print(len(data['data']))
# %%
