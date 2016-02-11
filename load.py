import gzip
import os
import numpy as np
import tensorflow as tf
from numpy import genfromtxt

def extract_data (filename):
    data = genfromtxt(filename, delimiter=',')
    return data

class DataSet(object):

  def __init__(self, inputs, labels, dtype=tf.float32):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    dtype = tf.as_dtype(dtype).base_dtype

    assert inputs.shape[0] == labels.shape[0], ('inputs.shape: %s labels.shape: %s' % (inputs.shape,
                                                                                       labels.shape))
    self._num_examples = inputs.shape[0]

    self._inputs = inputs
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def inputs(self):
    return self._inputs

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._inputs = self._inputs[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._inputs[start:end], self._labels[start:end]

def normalize (raw):
    mins = np.amin(raw,axis=0)
    offset = map (lambda x: x if x >= 0 else -x, mins)
    positive = np.add(raw, offset)
    maxs = np.amax(positive,axis=0)
    return np.true_divide(positive,maxs)

def read_data_sets(datad, validation, test, dtype=tf.float32, num=0):
  class DataSets(object):
    pass
  data_sets = DataSets()

  data = extract_data(dataf) if num == 0 else extract_data(dataf)[:num]
  # data = normalize(data)
  labels = extract_data(labelsf)[:,None] if num == 0 else  extract_data(labelsf)[:num,None]
  size = len(labels)
  validation_s = int(size * validation)

  test_s = int(size * test)

  training_s = size - validation_s - test_s

  print "Read: %d examples", size
  print "  Training set =   ", training_s
  print "  Validation set = ", validation_s
  print "  Test set =       ", test_s

  train_data = data[:training_s]
  train_labels = labels[:training_s]

  validation_data = data[training_s:training_s+validation_s]
  validation_labels = labels[training_s:training_s+validation_s]

  test_data = data[training_s+validation_s:training_s+validation_s+test_s]
  test_labels = labels[training_s+validation_s:training_s+validation_s+test_s]

  data_sets.train = DataSet(train_data, train_labels, dtype=dtype)
  data_sets.validation = DataSet(validation_data, validation_labels, dtype=dtype)
  data_sets.test = DataSet(test_data, test_labels, dtype=dtype)

  return data_sets
