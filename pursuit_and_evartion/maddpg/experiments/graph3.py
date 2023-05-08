#%%
import numpy as np
import matplotlib.pyplot as plt

temp = []
with open("test_TDerror.txt", "r") as f:
    for line in f.readlines():
        row = []
        line = line.replace('[', '')
        line = line.replace(']', '')
        toks = line.split(',')
        for tok in toks:
            row.append(tok)
        temp.append(row)

temp = temp[0]
sum = 0
for i in temp:
    str = i 
    sum += float(str)
sum = sum/len(temp)
# %%
