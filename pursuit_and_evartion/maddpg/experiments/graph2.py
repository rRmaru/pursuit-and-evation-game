#%%
import matplotlib.pyplot as plt
import numpy as np

x = range(30000)

y = []
for i in range(20000):
    y.append(0.6/20000*i+0.4)
    
print(y)

for i in range(10000):
    y.append(1)
    
plt.plot(x,y,color='red',label='beta')
plt.legend()
# %%
