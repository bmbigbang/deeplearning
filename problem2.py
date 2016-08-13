import os
from six.moves import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random as rnd

train_folders = ['notMNIST_large/A', 'notMNIST_large/B', 'notMNIST_large/C', 'notMNIST_large/D', 'notMNIST_large/E',
                 'notMNIST_large/F', 'notMNIST_large/G', 'notMNIST_large/H', 'notMNIST_large/I', 'notMNIST_large/J']
test_folders = ['notMNIST_small/A', 'notMNIST_small/B', 'notMNIST_small/C', 'notMNIST_small/D', 'notMNIST_small/E',
                'notMNIST_small/F', 'notMNIST_small/G', 'notMNIST_small/H', 'notMNIST_small/I', 'notMNIST_small/J']

# verify image data still looks good
dataset = pickle.load(open(train_folders[2] + '.pickle', 'rb'))
plt.imshow(dataset[22], interpolation='nearest', cmap=cm.brg)
plt.show()

# verify data is balanced accross all datasets by length comparison
dataset = pickle.load(open(train_folders[rnd.randint(0, 9)] + '.pickle', 'rb'))
print len(dataset)
dataset = pickle.load(open(train_folders[rnd.randint(0, 9)] + '.pickle', 'rb'))
print len(dataset)
dataset = pickle.load(open(train_folders[rnd.randint(0, 9)] + '.pickle', 'rb'))
print len(dataset)

# problem 5
duplicates = {}; norms = []
print dataset[0][0]
print len(dataset), len(dataset[0][0]), len(dataset[0][1])


# for i, x in enumerate(dataset):
#     norm = [np.linalg.norm(s) for s in x]
#     norms.append((norm, sum(norm)))
#
# for j, y in enumerate(dataset):
#     norm = norms[j][0]
#     for k, z in enumerate(dataset):
#         if j == k:
#             continue
#         diff = abs(norms[k][1] - sum(norm))
#         if diff > 0.2:
#             continue
#         diff2 = sum([abs(s-p) for s, p in zip(norms[k][0], norm)])
#         if diff2 < 0.5:
#             if j in duplicates:
#                 duplicates[j] += 1
#             else:
#                 duplicates[j] = 1

# for i, x in enumerate(dataset):
#     for j, y in enumerate(dataset):
#         if i == j:
#             continue
#         if (x==y).all():
#             if i in duplicates:
#                 duplicates[i] += 1
#             else:
#                 duplicates[i] = 1

# test tensordot
# print np.tensordot(dataset[0], dataset[1], axes=2)
# print sum([np.dot(i, j) for i, j in zip(dataset[0], dataset[1])])


