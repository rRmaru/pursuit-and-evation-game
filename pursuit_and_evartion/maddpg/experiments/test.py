#%%

import matplotlib.pyplot as plt
import pickle
# %%
x=range(0,30000,1000)
with open("learning_curves/None_agrewards.pkl", mode="rb") as f2:
    hoge2=pickle.load(f2)

good_agent = hoge2[3::4]
adv_agent=hoge2[::4]

list=[0.2605210420841683, 0.234, 0.336, 0.314, 0.488, 0.49, 0.5, 0.664, 0.686, 0.722, 0.722, 0.764, 0.776, 0.878, 1.048, 0.964, 0.81, 0.662, 0.878, 0.868, 0.664, 0.766, 0.814, 0.8, 0.802, 0.754, 0.932, 0.604, 0.584, 0.618, 0.586, 0.618, 0.656, 0.674, 0.608, 0.894, 0.732, 0.628, 0.7, 0.596, 0.676, 0.718, 0.68, 0.726, 0.888, 0.71, 0.854, 0.836, 0.972, 0.764, 0.704, 0.884, 0.796, 0.624, 0.644, 0.718, 0.71, 0.768, 0.676, 0.728]


fig = plt.figure()
ax1 = fig.add_subplot(212)
ln1 = ax1.plot(x,good_agent,color='red',label='pursuit')
#ax2 = fig.add_subplot(211)
#ln2 = ax2.plot(x,adv_agent,color='green',label='evasion')

ax3 = ax1.twinx()
ln3 = ax3.plot(x,list,color='blue',label='collision')

h1,l1 = ax1.get_legend_handles_labels()
h3,l3 = ax3.get_legend_handles_labels()
ax1.legend(h1+h3,l1+l3)

ax1.set_xlabel('episodes')
ax1.set_ylabel('reward')
ax1.grid(True)
ax3.set_ylabel('num')


# %%
plt.show()
# %%
