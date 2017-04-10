import random
import numpy as np
random.seed(10)
li = [1,2,3,4,5,6,7,8,9,0]
yi =[1,2,3,4,5,6,7,8,9,0]


z = zip(li, yi)
random.shuffle(z)
li_o, yi_o = zip(*z)
print li_o
print yi_o
x = list([1,2,3,4,5,6,7,8,9,0])
y = list([1,2,3,4,5,6,7,8,9,0])
np.random.seed(10)
shuffle_key = np.random.permutation(np.arange(len(y)))
print shuffle_key
x_shuffle = x[shuffle_key]
y_shuffle = y[shuffle_key]
print x_shuffle
print y_shuffle