#%%

import matplotlib.pyplot as plt
import pickle
import random
# %%
x=range(0,30000,1000)
with open("learning_curves/None_agrewards.pkl", mode="rb") as f2:
    hoge2=pickle.load(f2)

good_agent = hoge2[3::4]
adv_agent=hoge2[::4]

list=[0.2172172172172172, 0.429, 0.435, 0.461, 0.693, 1.884, 1.56, 1.013, 1.031, 1.131, 1.285, 1.338, 1.398, 1.304, 0.936, 0.801, 0.611, 0.52, 0.532, 0.455, 0.492, 0.437, 0.386, 0.434, 0.498, 0.47, 0.598, 0.434, 0.478, 0.576]


fig = plt.figure()
ax1 = fig.add_subplot(111)
ln1 = ax1.plot(x,good_agent,color='red',label='evation')
ax2 = fig.add_subplot(111)
ln2 = ax2.plot(x,adv_agent,color='green',label="pursuit")
#ax2 = fig.add_subplot(211)
#ln2 = ax2.plot(x,adv_agent,color='green',label='evasion')

ax3 = ax1.twinx()
ln3 = ax3.plot(x,list,color='blue',label='collision')

h1,l1 = ax1.get_legend_handles_labels()
h2,l2 = ax2.get_legend_handles_labels()
h3,l3 = ax3.get_legend_handles_labels()
ax1.legend(h1+h3,l1+l2+l3)

ax1.set_xlabel('episodes')
ax1.set_ylabel('reward')
ax1.grid(True)
ax3.set_ylabel('num')


# %%
plt.show()
# %%
