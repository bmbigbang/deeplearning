import numpy as np
import os
from scipy import ndimage
from six.moves import cPickle as pickle
from sklearn.preprocessing import normalize
import time

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

train_folders = ['notMNIST_large/A', 'notMNIST_large/B', 'notMNIST_large/C', 'notMNIST_large/D', 'notMNIST_large/E',
                 'notMNIST_large/F', 'notMNIST_large/G', 'notMNIST_large/H', 'notMNIST_large/I', 'notMNIST_large/J']
test_folders = ['notMNIST_small/A', 'notMNIST_small/B', 'notMNIST_small/C', 'notMNIST_small/D', 'notMNIST_small/E',
                'notMNIST_small/F', 'notMNIST_small/G', 'notMNIST_small/H', 'notMNIST_small/I', 'notMNIST_small/J']


def load_letter(folder, min_num_images):
    """Load the data for a single letter label."""
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
    print(folder)
    for image_index, image in enumerate(image_files):
        image_file = os.path.join(folder, image)
        try:
            image_data = (ndimage.imread(image_file).astype(float) -
                          pixel_depth / 2) / pixel_depth
            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[image_index, :, :] = image_data
        except IOError as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    num_images = image_index + 1
    dataset = dataset[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' %
                        (num_images, min_num_images))

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset


def maybe_pickle(data_folders, min_num_images_per_class, force=False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_letter(folder, min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)

    return dataset_names

train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)


# now merge and prune data

def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size, image_size)
    train_dataset, train_labels = make_arrays(train_size, image_size)
    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes

    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class + tsize_per_class
    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)
                # let's shuffle the letters to have random validation and training set
                np.random.shuffle(letter_set)
                if valid_dataset is not None:
                    valid_letter = letter_set[:vsize_per_class, :, :]
                    valid_dataset[start_v:end_v, :, :] = valid_letter
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class

                train_letter = letter_set[vsize_per_class:end_l, :, :]
                train_dataset[start_t:end_t, :, :] = train_letter
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    return valid_dataset, valid_labels, train_dataset, train_labels

train_size = 250000 # 210000
valid_size = 25000 # 12000
test_size = 18000 # 12000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(train_datasets,
                train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validating:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)

def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

# problem 5
np.seterr(divide='raise', invalid='print')


def duplicates(set1, set1_labels, s):
    dups = set([])
    norms1 = map(lambda x: np.tensordot(x, x, axes=2), set1)
    candidates = near_norms(norms1)
    print "starting direct comparisons with {0} dup candidates out of {1}".format(len(candidates), s)
    s = float(len(candidates))
    for i, x in enumerate(candidates):
        if i % 200 == 0:
            print "{0:.2f}%".format(100 * i / s)
        if x in dups:
            continue
        if abs(norms1[x]) <= 1e-7:
            dups.add(x)
            continue
        for j in candidates:
            if x <= j:
                break
            t = np.tensordot(set1[x], set1[j], axes=2)
            if (t / norms1[x]) > 0.948 and (t / norms1[j]) > 0.948:
                dups.add(x)
                break
    print "{0} dups found. Process Complete".format(len(dups))
    dups = list(dups)
    return np.delete(set1, dups, 0), np.delete(set1_labels, dups, 0)


def near_norms(s):
    cands = set(); total = float(len(s))
    print "starting norm candidates"
    for i, x in enumerate(s):
        if i % 2000 == 0:
            print "{0:.2f}%".format(100 * i / total)
        for j, y in enumerate(s):
            if i <= j:
                break
            if abs(x - y) <= 5e-5:
                cands.add(i)
                cands.add(j)
    print "100% completed norm candidates"
    return sorted(list(cands))


# s = duplicates(test_dataset[:200], test_dataset[50:250])
# print len(s)
# print s[:10]
# print len(s[0]), len(s[0][0])
# print test_dataset[:10]
# print len(test_dataset[0]), len(test_dataset[0][0])

size = train_size + test_size + valid_size
dataset = np.concatenate((train_dataset, test_dataset, valid_dataset))
labels = np.concatenate((train_labels, test_labels, valid_labels))
dataset, labels = duplicates(dataset, labels, size)

new_size = int(train_size*(float(len(dataset))/size))
print new_size, len(dataset)

train_dataset, train_labels = dataset[:new_size], labels[:new_size]
dataset, labels = dataset[new_size:], labels[new_size:]
test_dataset, test_labels = dataset[:(len(dataset)/2)], labels[:(len(labels)/2)]
valid_dataset, valid_labels = dataset[(len(dataset)/2):], labels[(len(labels)/2):]
# save randomized data

print len(train_dataset), len(train_labels)
print len(test_dataset), len(test_labels)
print len(valid_dataset), len(valid_labels)

pickle_file = 'notMNIST.pickle'

try:
    f = open(pickle_file, 'wb')
    save = {
        'train_dataset': train_dataset,
        'train_labels': train_labels,
        'valid_dataset': valid_dataset,
        'valid_labels': valid_labels,
        'test_dataset': test_dataset,
        'test_labels': test_labels,
    }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise
else:
    f.close()

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)
