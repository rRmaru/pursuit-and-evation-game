import numpy as np

from seed_test3 import test2

class test():
    def __init__(self, seed):
        self.seed = seed
        
    def make_rand(self):
       rand = [np.random.randint(0, 9) for _ in range(3)]
       return rand
   
    def call_rand(self):
        a = test2(self.seed)
        return a.make_rand()