import numpy as np
import re
import itertools
from collections import Counter
from sklearn import preprocessing
import codecs

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def line_to_words(line, dataset):
  if dataset == 'data/stsa.fine' or dataset == 'SST2':
    clean_line = clean_str_sst(line.strip())
  else:
    clean_line = clean_str(line.strip())
  words = clean_line.split(' ')
  words = words[1:]

  return words

def get_vocab(file_list, dataset=''):
  max_sent_len = 0
  word_to_idx = {}
  # word_to_idx['*PADDING*'] = 1
  # Starts at 2 for padding
  idx = 1

  for filename in file_list:
      f = open(filename, "r")
      for line in f:
          words = line_to_words(line, dataset)
          max_sent_len = max(max_sent_len, len(words))
          for word in words:
              if not word in word_to_idx:
                  word_to_idx[word] = idx
                  idx += 1

      f.close()

  return max_sent_len, word_to_idx


def clean_str_sst(string):
  """
  Tokenization/string cleaning for the SST dataset
  """
  string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
  string = re.sub(r"\s{2,}", " ", string)
  return string.strip().lower()

def load_data_all(dataset, padding=4, max_filter=5, padding_word="*PADDING*"):
  """
  Load all data, so that we feed it to CV.
  """

  f_names = [dataset]
  max_sent_len, word_to_idx = get_vocab(f_names, dataset)

  label = []
  data = []
  text_data = []

  pad_filter = max_filter - 1
  max_sent_len, word_to_idx = get_vocab(f_names, dataset)
  max_sent_len = max_sent_len + 2*pad_filter

  f_all = open(dataset, 'r', encoding = "ISO-8859-1")

  lines = filter(None, (line.rstrip() for line in f_all))

  for x in lines:
      label.append(int(x.split(' ')[0]))
      words = line_to_words(x[2:], dataset)
      text_data.append(words)
      sent = [word_to_idx[word] for word in words]
      num_padding = max_sent_len - len(sent) - pad_filter
      sent = [1]*max_filter + sent + [1]*num_padding
      data.append(sent)

  f_all.close()

  lb = preprocessing.LabelBinarizer()
  lb.fit(label)
  if len(lb.classes_)>2:
      bin_label = lb.transform(label)
  else:
      bin_label = np.array([[1,0] if l==0 else [0,1] for l in label])

  idx_to_word = {v: k for k, v in word_to_idx.items()}
  return word_to_idx, idx_to_word, label, np.array(data, dtype=np.int32), np.array(bin_label, dtype=np.int32), text_data

def load_data_pre_split(dataset, train_name='', test_name='', dev_name='', padding=4, max_filter=5, padding_word="*PADDING*"):
    """
    Load training data (dev/test optional).
    """
    train_name = dataset+'.train'
    test_name = dataset+'.test'
    dev_name = dataset+'.dev'

    if dataset=='data/TREC':
        dev_name = ''

    f_names = [train_name]
    f_names.append(test_name)
    f_names.append(dev_name)

    pad_filter = max_filter - 1

    max_sent_len, word_to_idx = get_vocab(f_names, dataset)
    max_sent_len = max_sent_len + 2*pad_filter

    dev = []
    dev_label = []
    train = []
    train_label = []
    test = []
    test_label = []

    files = []
    data = []
    data_label = []

    f_train = open(train_name, 'r')
    files.append(f_train)
    lines = f_train.readlines()
    for x in lines:
        train_label.append(int(x.split(' ')[0]))
        words = line_to_words(x[2:], dataset)
        sent = [word_to_idx[word] for word in words]
        num_padding = max_sent_len - len(sent) - pad_filter
        sent = [1]*max_filter + sent + [1]*num_padding
        train.append(sent)
    data.append(train)
    data_label.append(train_label)

    if not test_name == '':
        f_test = open(test_name, 'r')
        files.append(f_test)
        lines = f_test.readlines()
        for x in lines:
            test_label.append(int(x.split(' ')[0]))
            words = line_to_words(x[2:], dataset)
            sent = [word_to_idx[word] for word in words]
            num_padding = max_sent_len - len(sent) - pad_filter
            sent = [1]*max_filter + sent + [1]*num_padding
            test.append(sent)
        data.append(test)
        data_label.append(test_label)

    if not dev_name == '':
        f_dev = open(dev_name, 'r')
        files.append(f_dev)
        lines = f_dev.readlines()
        for x in lines:
            dev_label.append(int(x.split(' ')[0]))
            words = line_to_words(x[2:], dataset)
            sent = [word_to_idx[word] for word in words]
            num_padding = max_sent_len - len(sent) - pad_filter
            sent = [1]*max_filter + sent + [1]*num_padding
            dev.append(sent)
        data.append(dev)
        data_label.append(dev_label)

    f_train.close()
    if not test_name == '':
        f_test.close()
    if not dev_name == '':
        f_dev.close()

    all_labels = train_label+test_label+dev_label
    lb = preprocessing.LabelBinarizer()
    lb.fit(all_labels)
    if len(lb.classes_)>2:
        train_label = lb.transform(train_label)
        test_label = lb.transform(test_label)
        dev_label = lb.transform(dev_label)
    else:
        train_label = np.array([[1,0] if l==0 else [0,1] for l in train_label])
        test_label = np.array([[1,0] if l==0 else [0,1] for l in test_label])
        dev_label = np.array([[1,0] if l==0 else [0,1] for l in dev_label])

    idx_to_word = {v: k for k, v in word_to_idx.items()}
    return word_to_idx, idx_to_word, np.asarray(train, dtype=np.int32), np.array(train_label, dtype=np.int32), np.array(test, dtype=np.int32), np.array(test_label, dtype=np.int32), np.array(dev, dtype=np.int32), np.array(dev_label, dtype=np.int32), all_labels


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
