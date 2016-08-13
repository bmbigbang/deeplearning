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
import numpy as np
# .. svm method
# .. load data ..

X_train, X_test = np.reshape(train_dataset, (train_labels.shape[0], -1)), np.reshape(test_dataset, (test_labels.shape[0], -1))
# ..
# .. dimension reduction ..
pca = decomposition.RandomizedPCA(n_components=150, whiten=False)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# ..
# .. classification ..
clf = svm.SVC(C=5., gamma=0.012)
print X_train_pca.shape, X_test_pca.shape
clf.fit(X_train_pca, train_labels)
predicted = clf.predict(X_test_pca)
# ..
# .. predict on new images .

print sum([1 if i == j else 0 for i, j in zip(test_labels, predicted)])/float(len(test_labels))

# linear logistics method
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(solver='lbfgs')
clf.fit(X_train_pca, train_labels)
predicted = clf.predict(X_test_pca)

print sum([1 if i == j else 0 for i, j in zip(test_labels, predicted)])/float(len(test_labels))