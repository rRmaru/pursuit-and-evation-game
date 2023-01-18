""" Simplified  https://github.com/ray-project/ray/blob/master/rllib/execution/segment_tree.py
"""

import random

class SumTree:

  def __init__(self, capacity: int):
    #: 2のべき乗チェック
    assert capacity & (capacity - 1) == 0
    self.capacity = capacity
    self.values = [0 for _ in range(2 * capacity)]
  
  def __str__(self):
    return str(self.values[self.capacity:])
  
  def __setitem__(self, idx, val):
    idx = idx + self.capacity
    self.values[idx] = val

    current_idx = idx // 2
    while current_idx >= 1:
      idx_lchild = 2 * current_idx
      idx_rchild = 2 * current_idx + 1
      self.values[current_idx] = self.values[idx_lchild] + self.values[idx_rchild]
      current_idx //= 2

  def __getitem__(self, idx):
    idx = idx + self.capacity
    return self.value[idx]
  
  def sum(self):
    return self.values[1] 

  def sample(self, z=None):
    z = random.uniform(0, self.sum()) if z is None else z
    assert 0 <= z <= self.sum()
    
    current_idx = 1
    while current_idx < self.capacity:
      
      idx_lchild = 2 * current_idx
      idx_rchild = 2 * current_idx + 1

      #: 左子ノードよりzが大きい場合は右子ノードへ
      if z > self.values[idx_lchild]:
        current_idx = idx_rchild
        z = z -self.values[idx_lchild]
      else:
        current_idx = idx_lchild
    
    #: 見かけ上のインデックスにもどす
    idx = current_idx - self.capacity
    return idx

"""sumtree  = SumTree(capacity=4)
sumtree[0] = 4
sumtree[1] = 2
sumtree[2] = 1
sumtree[3] = 3

print("見かけ上の要素：", sumtree)
print("実際の要素：", sumtree.values)
print("総和", sumtree.sum())
print("累積和が6.5になるインデックス", sumtree.sample(z=6.5))

import numpy as np
import time 

N = 1000000
batch_size=512
indices = np.arange(N)
priorities = np.random.randint(0, 5, size=N)

a = time.time()
indeces = np.arange(N)
for _ in range(200):
    probs = priorities / priorities.sum()
    np.random.choice(indeces, size=batch_size, p=probs)
print("サンプリング時間", time.time() - a)

sumtree = SumTree(capacity=2**20)
for i, priority in enumerate(priorities):
    sumtree[i] = priority
    
a = time.time()
for _ in range(200):
    sampled = [sumtree.sample() for _ in range(512)]
print("サンプリング時間", time.time() - a)"""

