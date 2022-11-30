#%%

import matplotlib.pyplot as plt
import pickle
import random
x=range(0,30000,500)
with open("learning_curves/None_agrewards.pkl", mode="rb") as f2:
    hoge2=pickle.load(f2)

print(hoge2)
good_agent = hoge2[3::4]
adv_agent=hoge2[::4]
print(len(adv_agent))
#%%
list=[0.0781563126252505, 0.094, 0.182, 0.58, 0.806, 0.882, 0.962, 0.984, 1.004, 1.024, 1.016, 1.024, 1.008, 1.022, 1.014, 1.016, 1.018, 1.038, 1.032, 1.028, 1.01, 1.032, 1.024, 1.038, 1.026, 1.03, 1.01, 1.036, 1.028, 1.022, 1.0, 1.016, 1.016, 1.008, 1.01, 1.028, 1.016, 1.008, 0.98, 1.0, 0.994, 0.972, 0.996, 0.974, 0.98, 0.984, 0.982, 0.998, 0.988, 0.982, 1.002, 0.99, 1.004, 0.982, 0.992, 0.996, 0.988, 1.016, 1.026, 1.014]
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
list = [1.2942942942942943, 2.392, 4.796, 18.77, 19.751, 13.121, 12.946, 12.758, 14.27, 13.219]
x = range(0,10000,10)
plt.legend()
plt.plot(x,list,label="step len")

# %%
