#%%
import matplotlib.pyplot as plt
import pickle


# %%
with open("learning_curves/None_rewards.pkl", mode="rb") as f:
    hoge=pickle.load(f)
#%%
print(hoge)
# %%
x=range(0,60000,1000)
plt.plot(x, hoge, label="reward")
plt.legend()
plt.show()
# %%
with open("learning_curves/None_agrewards.pkl", mode="rb") as f2:
    hoge2=pickle.load(f2)

print(hoge2)
# %%
good_agent = hoge2[3::4]
print(good_agent)
print("========================")
adv_agent=hoge2[::4]
print(adv_agent)
# %%
fig=plt.figure(figsize=(6,4))

plt.plot(x,good_agent, color="blue", label="evation")
plt.plot(x,adv_agent, color="green", label="pursuit")
plt.legend()
plt.show()
# %%
