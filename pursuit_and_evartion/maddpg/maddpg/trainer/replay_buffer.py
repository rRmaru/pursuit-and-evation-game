import numpy as np
import random

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
        return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

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
    
    
class Prioritiy_ReplayBuffer(ReplayBuffer):
    def __init__(self, size):
        super().__init__(size)
        self.cnt = []
        self.alpha = 0
        
    def count(self, a):
        self.cnt.append(a)
        
    def make_indexTD(self, batch_size):
        order = []
        for i in range(len(self._storage)):
            obs_n, action, reward, obs_tp1, done, TDerror = self._storage[i]
            order.append([i, TDerror])
        order = sorted(order, reverse=True, key=lambda x:x[1])
        probability_D = []
        index = []
        for i in range(len(order)):
            D = 1/(i+1)
            probability_D.append(D)
            index.append(order[i][0])
        
        sum_D = sum([i**self.alpha for i in probability_D])
        
        p_D = [(D**self.alpha)/sum_D for D in probability_D]
        return [np.random.choice(index, p=p_D) for _ in range(batch_size)]
        