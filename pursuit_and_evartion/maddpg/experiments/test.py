#%%

import matplotlib.pyplot as plt
import pickle
import random
x=range(0,30000,500)
with open("learning_curves/None_agrewards.pkl", mode="rb") as f2:
    hoge2=pickle.load(f2)

good_agent = hoge2[3::4]
adv_agent=hoge2[::4]
print(len(good_agent))
print(len(adv_agent))
#%%
list=[0.06613226452905811, 0.06, 0.218, 0.602, 0.66, 0.642, 0.878, 0.862, 0.886, 0.948, 0.918, 0.77, 0.722, 0.738, 0.74, 0.668, 0.666, 0.656, 0.662, 0.71, 0.542, 0.592, 0.732, 0.786, 0.774, 0.756, 0.712, 0.778, 0.722, 0.694, 0.75, 0.802, 0.906, 0.918, 0.884, 0.85, 0.846, 0.914, 0.932, 0.886, 0.876, 0.874, 0.884, 0.89, 0.858, 0.874, 0.83, 0.764, 0.73, 0.734, 0.69, 0.706, 0.712, 0.718, 0.738, 0.75, 0.772, 0.824, 0.808, 0.778]
print(len(list))
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
list = [0.2665330661322645, 0.31, 0.148, 0.138, 0.378, 0.432, 0.478, 0.742, 0.754, 0.744, 0.846, 0.91, 0.92, 0.738, 0.558, 0.744, 0.896, 0.958, 0.964, 0.924, 0.892, 0.916, 0.95, 0.922, 0.904, 0.906, 0.926, 0.946, 0.912, 0.914, 0.804, 0.866, 0.864, 0.834, 0.842, 0.862, 0.872, 0.846, 0.858, 0.876, 0.842, 0.818, 0.786, 0.766, 0.788, 0.816, 0.8, 0.768, 0.774, 0.762, 0.796, 0.784, 0.776, 0.77, 0.79, 0.756, 0.758, 0.746, 0.744, 0.768]
x = range(0,10000,10)
plt.legend()
plt.plot(x,list,label="step len")

# %%