import numpy as np

class test2():
    def __init__(self, seed):
        self.seed = seed
        
    def make_rand(self):
        rand = [np.random.randint(0, 9) for _ in range(3)]
        return rand