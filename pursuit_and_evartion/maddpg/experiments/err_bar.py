#%%
import numpy as np
import matplotlib.pyplot as plt
import sys
alpha0 = []
with open("rewards_alpha0_1.txt", "r") as f:
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
with open("rewards_alpha1_1.txt", "r") as f:
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
with open("rewards_alpha0_2_0418.txt", "r") as f:
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
alpha0.pop()
alpha0 = alpha0 + alpha0_3
print(len(alpha0))

alpha0_4 = []
with open("rewards_alpha1_2_0417.txt", "r") as f:
    for line in f.readlines():#行をすべて読み込んで一行ずつfor文で回す
        row = []
        line = line.replace('[', '')
        line = line.replace(']', '')
        toks = line.split(',')#行をカンマで分割
        for tok in toks:
            num = float(tok)
            row.append(num)#行に保存
        alpha0_4.append(row)#行をnumsに保存
print(len(alpha0_4[0]))
alpha0_4 = alpha0_4[0]
alpha0_1.pop()
alpha0_1 = alpha0_1 + alpha0_4
print(len(alpha0_1))

#%%
save_rate = 500
rewards0 = []
rewards1 = []
rewards2 = []

y_err= []
for i in range(int(len(alpha0)/save_rate)):
    rewards0.append(np.mean(alpha0_1[i*save_rate:i*save_rate+1001]))
    #rewards1.append(np.mean(alpha0_1[i*save_rate:i*save_rate+1001]))
    y_err.append(np.std(np.array(alpha0_1[i*save_rate:i*save_rate+1001])))
    
x = range(0, 30000, save_rate)
fix, ax = plt.subplots()
ax.errorbar(x, rewards0, yerr=y_err, color='red', label='alpha=0')



ax1.legend()

ax1.set_xlabel('episode')
ax1.set_ylabel('reward')
ax1.grid(True)

ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

# %%
plt.plot(x, y_err, color='blue')
plt.ylim(0, 180)

plt.show()
# %%
