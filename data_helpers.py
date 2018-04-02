import numpy as np
import re
import itertools
from collections import Counter



def load_data_and_labels(source_file, target_file, labels_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    source_examples = list(open(source_file, "r", encoding='utf-8').readlines())
    source_examples = [s.strip() for s in source_examples]
    target_examples = list(open(target_file, "r", encoding='utf-8').readlines())
    target_examples = [s.strip() for s in target_examples]
    labels_list = list(open(labels_file, "r", encoding='utf-8').readlines())
    labels_list = [s.strip() for s in labels_list]
    labels_dict = dict(zip(labels_list, range(len(labels_list))))
    # sorted_tmp = sorted(labels_dict.items(), key=lambda x: x[1])
    x_text = source_examples
    y = target_examples
    return [x_text, y, labels_dict]

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