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

list=[0.6916916916916916, 8.071, 18.604, 12.652, 7.264, 6.119, 5.57, 5.014, 4.595, 3.905, 3.901, 3.575, 4.089, 3.943, 4.536, 3.92, 4.952, 5.739, 7.529, 6.889, 5.414, 4.671, 4.47, 4.114, 3.759, 3.631, 3.531, 3.49, 3.255, 3.383]

fig = plt.figure()
ax1 = fig.add_subplot(111)
ln1 = ax1.plot(x,good_agent,color='red',label='evation')
ax2 = fig.add_subplot(111)
ln2 = ax2.plot(x,adv_agent,color='green',label="pursuit")

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