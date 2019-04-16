import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def split_train_test(df, class_name, test_size = 0.3):
    X = df.drop(class_name, axis = 1)
    y = df[class_name]
    size = len(X)
    test_index = df.index.isin(np.sort(np.random.choice(range(len(X)), int(np.round(size*test_size)), replace = False)))
    X_train = df[test_index]
    X_test = df[~test_index]
    y_train = y[test_index]
    y_test = y[~test_index]
    return X_train, X_test, y_train, y_test

def split_dataset(df, n_datasets, sample_size, replace = True):
    if replace and (len(df) < n_datasets * sample_size):
        raise("For sampling with replace, n_datasets * sample_size must be less than df size (%d * %d < %d)" % (n_datasets, sample_size, len(df)))
    
    datasets = []
    for i in range(n_datasets):
        datasets.append(df[__sample_dataset(df, sample_size, replace)])
    
    return datasets

def holdout(df, class_name, test_size = 0.3):
    return split_train_test(df, class_name, test_size)

# todo: check len(df)/k is not int
# todo: finish this method
def kfold(df, k):
    sample_size = int(np.floor(len(df)/k))
    return df__sample_dataset(df, sample_size, rest = len(df)%k, replace = False)
    
def __sample_dataset(df, sample_size, replace, rest = 0):
    return df.index.isin(np.sort(np.random.choice(range(len(df)), sample_size, replace = replace)))