#%%
import numpy as np
import matplotlib.pyplot as plt
import sys
alpha0 = []
with open("rewards_alpha1_1.txt", "r") as f:
    for line in f.readlines():#行をすべて読み込んで一行ずつfor文で回す
        row = []
        line = line.replace('[', '')
        line = line.replace(']', '')
        toks = line.split(',')#行をカンマで分割
        for tok in toks:
            num = float(tok)
            row.append(num)#行に保存
        alpha0.append(row)#行をnumsに保存
print(len(alpha0[0]))
alpha0 = alpha0[0]

alpha0_1 = []
with open("rewards_alpha0_1.txt", "r") as f:
    for line in f.readlines():#行をすべて読み込んで一行ずつfor文で回す
        row = []
        line = line.replace('[', '')
        line = line.replace(']', '')
        toks = line.split(',')#行をカンマで分割
        for tok in toks:
            num = float(tok)
            row.append(num)#行に保存
        alpha0_1.append(row)#行をnumsに保存
print(len(alpha0_1[0]))
alpha0_1 = alpha0_1[0]

alpha0_3 = []
with open("rewards_alpha0_3.txt", "r") as f:
    for line in f.readlines():#行をすべて読み込んで一行ずつfor文で回す
        row = []
        line = line.replace('[', '')
        line = line.replace(']', '')
        toks = line.split(',')#行をカンマで分割
        for tok in toks:
            num = float(tok)
            row.append(num)#行に保存
        alpha0_3.append(row)#行をnumsに保存
print(len(alpha0_3[0]))
alpha0_3 = alpha0_3[0]



# %%
save_rate = 500
rewards0 = []
rewards1 = []
for i in range(int(len(alpha0)/save_rate)):
    rewards0.append(np.mean(alpha0[i*save_rate:i*save_rate+1001]))
    rewards1.append(np.mean(alpha0_1[i*save_rate:i*save_rate+1001]))
    
x = range(0, 20000, save_rate)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ln1 = ax1.plot(x, rewards0, color='blue', label='alpha=0.1')

ax2 = fig.add_subplot(111)
ln2 = ax2.plot(x, rewards1, color='red', label='alpha=0')

ax1.legend()

ax1.set_xlabel('episode')
ax1.set_ylabel('reward')
ax1.grid(True)

# %%
