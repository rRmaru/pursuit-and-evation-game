#%%

import matplotlib.pyplot as plt
import pickle
import random
# %%
x=range(0,30000,500)
with open("learning_curves/None_agrewards.pkl", mode="rb") as f2:
    hoge2=pickle.load(f2)

print(hoge2)
good_agent = hoge2[4::5]
adv_agent=hoge2[::5]
a = hoge2[0]
#%%
list=[]
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
ax1.set_ylim(-1500, 1500)
ax1.grid(True)
ax3.set_ylabel('num')
ax3.set_ylim(0.0,25.0)


# %%
list = [1.2942942942942943, 2.392, 4.796, 18.77, 19.751, 13.121, 12.946, 12.758, 14.27, 13.219]
x = range(0,10000,10)
plt.legend()
plt.plot(x,list,label="step len")

# %%
