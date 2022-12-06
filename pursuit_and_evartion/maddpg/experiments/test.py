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
list=[0.04208416833667335, 0.04, 0.248, 0.406, 0.462, 0.804, 0.874, 0.97, 1.016, 0.992, 1.004, 1.004, 1.014, 1.01, 1.01, 1.002, 0.984, 1.014, 1.01, 1.02, 1.028, 1.018, 1.034, 1.042, 0.972, 0.712, 0.646, 0.582, 0.544, 0.608, 0.606, 0.638, 0.726, 0.788, 0.832, 0.86, 0.872, 0.832, 0.83, 0.902]
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
list = [0.04208416833667335, 0.04, 0.248, 0.406, 0.462, 0.804, 0.874, 0.97, 1.016, 0.992, 1.004, 1.004, 1.014, 1.01, 1.01, 1.002, 0.984, 1.014, 1.01, 1.02, 1.028, 1.018, 1.034, 1.042, 0.972, 0.712, 0.646, 0.582, 0.544, 0.608, 0.606, 0.638, 0.726, 0.788, 0.832, 0.86, 0.872, 0.832, 0.83, 0.902]
x = range(0,10000,10)
plt.legend()
plt.plot(x,list,label="step len")

# %%