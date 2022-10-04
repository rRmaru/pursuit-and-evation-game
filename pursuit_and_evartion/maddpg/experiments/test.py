#%%
import gym
env = gym.make("MountainCar-v0")

# %%
from gym import envs
envids = [spec.id for spec in envs.registry.all()]

observation = env.reset()

# %%
env.render()
env.close()
# %%
