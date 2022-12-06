#%%

import matplotlib.pyplot as plt
import pickle
import random
x=range(0,20000,500)
with open("learning_curves/None_agrewards.pkl", mode="rb") as f2:
    hoge2=pickle.load(f2)

good_agent = hoge2[3::4]
adv_agent=hoge2[::4]
print(len(good_agent))
print(len(adv_agent))
list=[0.21442885771543085, 0.212, 0.224, 0.348, 0.552, 0.812, 0.956, 0.978, 1.004, 1.004, 0.998, 1.014, 1.01, 1.01, 1.004, 0.984, 1.004, 1.002, 1.0, 1.002, 0.996, 1.002, 0.988, 0.994, 0.982, 0.98, 0.984, 0.958, 0.946, 0.974, 0.964, 0.998, 0.996, 0.98, 0.954, 0.97, 0.95, 0.874, 0.9, 0.9]
#%%
fig = plt.figure()
ax1 = fig.add_subplot(111)
ln1 = ax1.plot(x,good_agent,color='red',label='evation')
ax2 = fig.add_subplot(111)
ln2 = ax2.plot(x,adv_agent,color='blue',label="pursuit")

ax3 = ax1.twinx()
ln3 = ax3.plot(x,list,color='green',label='collision')

h1,l1 = ax1.get_legend_handles_labels()
h3,l3 = ax3.get_legend_handles_labels()
ax1.legend(h1+h3,l1+l3)

ax1.set_xlabel('episodes')
ax1.set_ylabel('reward')
ax1.set_ylim(-150, 150)
ax1.grid(True)
ax3.set_ylabel('num')
ax3.set_ylim(0.0,25.0)


# %%
list = [0.32064128256513025, 0.302, 0.27, 0.586, 0.578, 0.594, 0.756, 0.938, 0.986, 1.0, 0.996, 0.972, 0.98, 0.98, 0.95, 0.98, 0.968, 0.998, 0.988, 1.012, 1.014, 1.048, 1.04, 0.996, 0.982, 0.932, 0.984, 0.96, 0.972, 0.94, 0.966, 0.96, 0.938, 0.974, 0.964, 0.974, 0.95, 0.962, 0.976, 0.952]
plt.legend()
plt.plot(x,list,label="step len")

# %%