from random import randint
import numpy as np
from scipy import convolve
from math import exp
from matplotlib import pyplot as plt

x = np.arange(-3, 3, 0.1)
y = map(lambda i: exp(-i**2/1.2), x)
z = [randint(0,10)/10.0 for j in x]
plt.plot(x, convolve(y, z, mode='same'))
plt.show()

from six.moves import cPickle as pickle
pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)

from sklearn import svm, decomposition
# .. svm method
# .. load data ..

X_train, X_test = np.reshape(train_dataset, (train_labels.shape[0], -1)), np.reshape(test_dataset, (test_labels.shape[0], -1))
# ..
# .. dimension reduction ..
pca = decomposition.RandomizedPCA(n_components=150, whiten=False)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

from matplotlib import cm
a = np.reshape(pca.components_[0], train_dataset[0].shape)

plt.imshow(a, interpolation='nearest', cmap=cm.brg)
plt.show()
