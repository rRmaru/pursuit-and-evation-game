import numpy as np
import os

from seed_test2 import test

def fix_seed(seed):
    np.random.seed(seed)
    
def make_rand():
    rand = [np.random.randint(0, 9) for i in range(3)]
    return rand
    
if __name__ == "__main__":
    os.environ['PYTHONHASHSEED'] = '0'
    SEED = 42
    fix_seed(SEED)
    print(make_rand())
    fix_seed(SEED)
    print(make_rand())
    
    a = test(SEED)
    fix_seed(SEED)
    print(a.call_rand())
    fix_seed(SEED)
    print(a.call_rand())