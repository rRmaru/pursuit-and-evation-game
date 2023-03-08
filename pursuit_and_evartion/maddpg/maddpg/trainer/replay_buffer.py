import ipdb as pdb

import time
import numpy as np
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

class ReplayBuffer(object):
    def __init__(self, size):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def add(self, obs_t, action, reward, obs_tp1, done, TDerror):           #obs, action, reward, obs_next, done
        data = (obs_t, action, reward, obs_tp1, done, TDerror)       #tupleを作成
        
        if self._next_idx >= len(self._storage):        #_next_indexの大きさがstorageの大きさと同じかそれ以上
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize       #indexをひとつ追加するまた、maxsizeを超えると二週目を開始する
        
    def _encode_sample(self, idxes):                #sample indexで呼び出される関数
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]        #random番目のデータを取り出す
            obs_t, action, reward, obs_tp1, done, TDerror = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def make_index(self, batch_size):       #randomにバッファの中から取り出す（バッチサイズの数だけ）PERではこの部分を変更したい
        selected_p = np.ones(batch_size)
        selected_p = selected_p/1e5
        rand =  [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return rand, selected_p

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._encode_sample(idxes)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))
        return self._encode_sample(idxes)

    def collect(self):
        return self.sample(-1)
    
    
class Priority_ReplayBuffer(ReplayBuffer):
    def __init__(self, size):
        super().__init__(size)
        self.cnt = []
        self.alpha = 0.1
        
    def count(self, a):
        self.cnt.append(a)
        
    def make_indexTD(self, batch_size):
        a = time.time()
        order = []
        for i in range(len(self._storage)):
            obs_n, action, reward, obs_tp1, done, TDerror = self._storage[i]
            order.append([i, TDerror])
        order = sorted(order, reverse=True, key=lambda x:x[1])
        probability_D = []
        idx = []
        for i in range(len(order)):
            D = 1/(i+1)
            probability_D.append(D)
            idx.append(order[i][0])
        c = time.time()
        sum_D = sum([i**self.alpha for i in probability_D])
        
        p_D = [(D**self.alpha)/sum_D for D in probability_D]
        rand = np.random.choice(idx, p=p_D, size=batch_size)
        selected_p = []
        for i in rand:
            selected_p.append(p_D[i])
            
        return rand, selected_p
    
    def TD_update(self, obs, act, rew, obs_next, done, TDerror):
        for i in range(len(obs)):
            self._storage[i] = (obs[i], act[i], rew[i], obs_next[i], done[i], TDerror[i])
    
    
                