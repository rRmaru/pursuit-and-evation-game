import numpy as np
import random
import tensorflow as tf

def fix_seed(seed):
  #randm
  random.seed(seed)
  #Numpy
  np.random.seed(seed)
  #Tensorflow
  tf.random.set_seed(seed)
  
SEED = 42
fix_seed(SEED)