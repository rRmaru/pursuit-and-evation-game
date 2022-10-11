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

list=[0.23023023023023023, 0.479, 0.407, 0.398, 0.587, 0.908, 1.426, 1.952, 1.493, 1.353, 1.612, 1.522, 1.468, 1.325, 1.369, 1.218, 1.102, 1.137, 1.193, 1.163, 1.126, 1.062, 1.046, 1.086, 1.04, 0.936, 1.06, 1.023, 0.916, 0.987]


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
