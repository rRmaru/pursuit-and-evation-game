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

list=[0.5745745745745746, 1.103, 12.922, 10.357, 12.609, 11.251, 5.766, 3.999, 4.918, 5.413, 8.809, 9.975, 8.984, 7.365, 5.942, 5.388, 6.399, 6.867, 6.597, 6.268, 6.115, 5.129, 5.578, 5.357, 5.802, 6.35, 6.292, 5.963, 6.275, 6.703]

fig = plt.figure()
ax1 = fig.add_subplot(111)
ln1 = ax1.plot(x,good_agent,color='red',label='evation')
ax2 = fig.add_subplot(111)
ln2 = ax2.plot(x,adv_agent,color='green',label="pursuit")

ax3 = ax1.twinx()
ln3 = ax3.plot(x,list,color='blue',label='collision')

h1,l1 = ax1.get_legend_handles_labels()
h3,l3 = ax3.get_legend_handles_labels()
ax1.legend(h1+h3,l1+l3)

ax1.set_xlabel('episodes')
ax1.set_ylabel('reward')
ax1.set_ylim(-300, 300)
ax1.grid(True)
ax3.set_ylabel('num')
ax3.set_ylim(0,18)


# %%
plt.show()