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

list=[0.7377377377377378, 1.404, 24.413, 6.486, 3.381, 2.392, 2.696, 6.28, 15.446, 14.966, 10.935, 4.668, 3.942, 9.895, 12.048, 7.561, 8.343, 8.341, 6.789, 5.936, 6.694, 6.539, 6.786, 6.825, 6.19, 6.645, 6.159, 6.22, 6.513, 7.111]


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