#%%
import matplotlib.pyplot as plt
import pickle


# %%
with open("learning_curves/None_rewards.pkl", mode="rb") as f:
    hoge=pickle.load(f)
#%%
print(hoge)
# %%
x=range(0,20000,1000)
plt.plot(x, hoge, label="test")
plt.legend()
plt.show()
# %%
