#%%

import matplotlib.pyplot as plt
import pickle
import random
x=range(0,15000,250)
with open("learning_curves/permaddpg_alpha0.5_agrewards.pkl", mode="rb") as f2:
    hoge2=pickle.load(f2)

good_agent = hoge2[3::4]
adv_agent=hoge2[::4]
print(len(good_agent))
print(len(adv_agent))
list=[0.1285140562248996, 0.116, 0.336, 0.292, 0.372, 0.308, 0.724, 0.756, 0.936, 0.952, 0.984, 0.968, 0.988, 0.976, 0.98, 1.004, 0.984, 0.976, 0.964, 0.968, 0.956, 0.98, 0.992, 0.984, 0.98, 0.96, 0.996, 0.988, 0.952, 0.984, 0.98, 1.0, 1.008, 1.024, 1.028, 0.996, 0.816, 0.852, 0.804, 0.804, 0.744, 0.828, 0.912, 0.992, 0.992, 1.008, 1.012, 0.972, 1.02, 1.008, 0.996, 0.956, 0.908, 0.868, 0.864, 0.872, 0.872, 0.868, 0.868, 0.892]
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
ax3.set_ylabel('collision_num')
ax3.set_ylim(0.0,25.0)



# %%
list = [0.1843687374749499, 0.382, 0.308, 0.368, 0.42, 0.572, 0.47, 0.574, 0.64, 0.822, 0.944, 0.986, 0.932, 0.952, 0.908, 0.832, 0.862, 0.854, 0.864, 0.844, 0.864, 0.892, 0.886, 0.908, 0.892, 0.934, 0.942, 0.96, 0.944, 0.95, 0.934, 0.968, 0.942, 0.942, 0.928, 0.946, 0.972, 0.974, 0.96, 0.948]
plt.legend()
plt.plot(x,list,label="step len")

# %%