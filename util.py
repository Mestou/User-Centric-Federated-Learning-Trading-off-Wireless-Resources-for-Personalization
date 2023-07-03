from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
import torch
import os
from tensorflow.keras.datasets import mnist,fashion_mnist,cifar10
from extra_keras_datasets import emnist
import re
import shutil
import string
import tensorflow_datasets as tfds





def label_swap(array,perm):
    modified = array.copy()
    for i in range(0,len(perm)):
        modified[np.where(array == i)] = perm[i]
    return modified

def loadMNIST(nodes):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255., x_test / 255.
    num_classes = 10
    x_shape = x_train.shape[1:]
    ind_class = [np.where(y_train == i) for i in range(0, num_classes)]
    ind_class_te = [np.where(y_test == i) for i in range(0, num_classes)]
    class_user = [[0,1],[2,3],[4,5],[2,0],[8,9],[0,2],[1,3],[4,6],[5,7],[0,1,2,3]]
    samples_user = [1000, 2000, 500, 2100, 300, 300, 100, 150, 100, 150]
    train_X, train_Y, test_X, test_Y = [], [], [], []
    for i in range(0, nodes):
        ind_class_user = np.concatenate([ind_class[i][0] for i in class_user[i]], axis=0)
        indices = np.random.choice(ind_class_user, size=samples_user[i], replace=False)
        train_X.append(x_train[indices, :, :])
        train_Y.append(y_train[indices])
        ind_class_user_te = np.concatenate([ind_class_te[i][0] for i in class_user[i]], axis=0)
        test_X.append(x_test[ind_class_user_te, :, :])
        test_Y.append(y_test[ind_class_user_te])
    fracs = [samples_user / np.sum(samples_user)]
    fracs = np.zeros((nodes, nodes), dtype=float)
    for i in range(0, nodes):
        for j in range(0, nodes):
            fracs[i, j] = samples_user[j] / samples_user[i]
    train_Y = [tf.one_hot(train_Y[i], depth=num_classes).numpy() for i in range(0, nodes)]
    test_Y = [tf.one_hot(test_Y[i], depth=num_classes).numpy() for i in range(0, nodes)]
    return (train_X, train_Y), (test_X , test_Y),  x_shape, num_classes, fracs

def loadEMNIST(nodes):

    (x_train, y_train), (x_test, y_test) = emnist.load_data(type="byclass")
    x_train, x_test = x_train / 255., x_test / 255.

    x_train, x_test =np.expand_dims(x_train,axis=3),np.expand_dims(x_test,axis=3)

    num_classes = 62
    x_shape=x_train.shape[1:]

    train_frac = 0.5

    samples_user=[]
    train_X, train_Y, test_X, test_Y = [], [], [], []
    n_classes = np.max(y_train)+1
    idcs = np.random.permutation(x_train.shape[0])
    print(idcs)
    train_idcs= idcs[:20000]
    alpha=0.4                        ######## sets the dirichlet parameter VALUE ######
    label_distribution = np.random.dirichlet([alpha]*nodes, n_classes)
    class_idcs = [np.argwhere(y_train[train_idcs]==y).flatten()
           for y in range(n_classes)]

    client_idcs = [[] for _ in range(nodes)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [train_idcs[np.concatenate(idcs)] for idcs in client_idcs]
    for idc in client_idcs:
        n_train = int(len(idc) * train_frac)
        n_eval = len(idc) - n_train
        idc_train, idc_eval = torch.utils.data.random_split(idc, [n_train, n_eval])

        train_X.append(x_train[idc_train, :, :])
        train_Y.append(y_train[idc_train])
        samples_user.append(n_train)
        test_X.append(x_train[idc_eval, :, :])
        test_Y.append(y_train[idc_eval])
    print("Nodes dataset sizes : ")
    print(samples_user)
    fracs = np.zeros((nodes, nodes), dtype=float)
    for i in range(0, nodes):
        for j in range(0, nodes):
            fracs[i, j] = samples_user[j] / samples_user[i]


                                        #### uncomment for applying rotation transformation on images (20 users, 4 groups) ####
    # for i in range(0, 5):
    #     train_X[i] = np.asarray(tf.image.rot90(train_X[i]))
    #     test_X[i] = np.asarray(tf.image.rot90(test_X[i]))
    #
    # for i in range(5, 10):
    #     train_X[i] = np.asarray(tf.image.rot90(tf.image.rot90(train_X[i])))
    #     test_X[i] = np.asarray(tf.image.rot90(tf.image.rot90(test_X[i])))
    #
    # for i in range(10, 15):
    #     train_X[i] = np.asarray(tf.image.rot90(tf.image.rot90(tf.image.rot90(train_X[i]))))
    #     test_X[i]= np.asarray(tf.image.rot90(tf.image.rot90(tf.image.rot90(test_X[i]))))
    #
    # for i in range(15, 20):
    #     train_X[i] = np.asarray(tf.image.rot90(tf.image.rot90(tf.image.rot90(tf.image.rot90(train_X[i])))))
    #     test_X[i]= np.asarray(tf.image.rot90(tf.image.rot90(tf.image.rot90(tf.image.rot90(test_X[i])))))


                                            #### uncomment for label swap experiment ####
    # print(samples_user)
    # p = np.random.permutation(62)
    # for i in range(0,25):
    #     train_Y[i]=label_swap(train_Y[i],p)
    #     test_Y[i]=label_swap(test_Y[i],p)
    # p = np.random.permutation(62)
    # for i in range(25, 50):  ##
    #     train_Y[i]=label_swap(train_Y[i],p)
    #     test_Y[i]=label_swap(test_Y[i],p)
    # p = np.random.permutation(62)
    # for i in range(50, 75):
    #     train_Y[i]=label_swap(train_Y[i],p)
    #     test_Y[i]=label_swap(test_Y[i],p)
    # p = np.random.permutation(62)
    # for i in range(75, 100):
    #     train_Y[i]=label_swap(train_Y[i],p)
    #     test_Y[i]=label_swap(test_Y[i],p)

                                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    train_Y = [tf.one_hot(train_Y[i], depth=num_classes) for i in range(0, nodes)]
    test_Y = [tf.one_hot(test_Y[i], depth=num_classes) for i in range(0, nodes)]
    return (train_X, train_Y), (test_X , test_Y),  x_shape, num_classes, fracs, np.asarray(samples_user)


def loadCIFAR10(nodes):

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    #x_train=np.sum(x_train/3,axis=3,keepdims=True)
    #x_test = np.sum(x_test / 3, axis=3, keepdims=True)
    y_train, y_test = np.squeeze(y_train), np.squeeze(y_test)
    x_train, x_test = x_train / 255., x_test / 255.
    num_classes = 10
    x_shape = x_train.shape[1:]
    samples_user = np.random.randint(99, 100, size=nodes)
    samples_user = np.ceil(50000 * samples_user / np.sum(samples_user)).astype(int)
    class_user = np.arange(0, 10)
    ind_class = [np.where(y_train == i) for i in range(0, 10)]
    train_X, train_Y, test_X, test_Y = [], [], [], []
    assigned, client_idcs, val_idcs = [], [], []  # Used to track assigned samples and avoid overlapping datasets
    for i in range(0, nodes):
        ind_class_user = np.concatenate([ind_class[i][0] for i in class_user], axis=0)
        ind_class_user = np.setdiff1d(ind_class_user, np.array(assigned), assume_unique=True)
        indices = np.random.choice(ind_class_user, size=samples_user[i], replace=False)
        assigned.append(indices)
        client_idcs.append(indices)
        val_idcs.append(np.arange(0, 1000))
    for i in range(0,nodes):
        train_X.append(x_train[client_idcs[i], :, :])
        train_Y.append(y_train[client_idcs[i]])
        test_X.append(x_test[val_idcs[i], :, :])
        test_Y.append(y_test[val_idcs[i]])
    print("Nodes dataset sizes : ")
    print(samples_user)

                                #### comment/uncomment for label swap experiment ####

    p = np.random.permutation(10)
    for i in range(0,5):
        train_Y[i]=label_swap(train_Y[i],p)
        test_Y[i]=label_swap(test_Y[i],p)
    p = np.random.permutation(10)
    for i in range(5, 10):  ##
        train_Y[i]=label_swap(train_Y[i],p)
        test_Y[i]=label_swap(test_Y[i],p)
    p = np.random.permutation(10)
    for i in range(10, 15):
        train_Y[i]=label_swap(train_Y[i],p)
        test_Y[i]=label_swap(test_Y[i],p)
    p = np.random.permutation(10)
    for i in range(15, 20):
        train_Y[i]=label_swap(train_Y[i],p)
        test_Y[i]=label_swap(test_Y[i],p)
                                                        ####################################

    fracs = np.zeros((nodes,nodes),dtype= float)
    for i in range(0,nodes):
        for j in range(0,nodes):
            fracs[i,j] = samples_user[j]/samples_user[i]

    train_Y = [tf.one_hot(train_Y[i], depth=num_classes).numpy() for i in range(0, nodes)]
    test_Y = [tf.one_hot(test_Y[i], depth=num_classes).numpy() for i in range(0, nodes)]
    return (train_X, train_Y), (test_X , test_Y),  x_shape, num_classes, fracs,samples_user


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def loadSentiment(nodes):

    batch_size = 1
    seed = 42

    raw_train_ds = tf.keras.utils.text_dataset_from_directory(
        './aclImdb/train',
        batch_size=batch_size,
        seed=seed)
    raw_test_ds = tf.keras.utils.text_dataset_from_directory(
        './aclImdb/test',
        batch_size=batch_size)

    max_features = 10000
    sequence_length = 3000


    vectorize_layer = tf.keras.layers.TextVectorization(
        standardize=custom_standardization,
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length)

    train_text = raw_train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(train_text)

    def vectorize_text(text, label):
        text = tf.expand_dims(text, -1)
        return vectorize_layer(text), label

    train_ds = raw_train_ds.map(vectorize_text)
    test_ds = raw_test_ds.map(vectorize_text)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    x_train,y_train = dataset_to_numpy(train_ds)
    x_test,y_test = dataset_to_numpy(test_ds)
    x_train = np.concatenate((x_train,x_test),axis=0)
    y_train = np.concatenate((y_train,y_test))
    y_train = y_train.astype('float32')

    num_classes = 4

    x_shape = x_train.shape[1:]

    train_frac = 0.90

    samples_user = []
    train_X, train_Y, test_X, test_Y = [], [], [], []
    n_classes = np.max(y_train).astype('int') + 1
    idcs = np.random.permutation(x_train.shape[0])
    print(idcs)
    train_idcs = idcs[:16000]

    alpha = 0.4  ######## sets the dirichlet parameter VALUE ######
    label_distribution = np.random.dirichlet([alpha] * nodes, n_classes)
    class_idcs = [np.argwhere(y_train[train_idcs] == y).flatten()
                  for y in range(n_classes)]

    client_idcs = [[] for _ in range(nodes)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [train_idcs[np.concatenate(idcs)] for idcs in client_idcs]
    for idc in client_idcs:
        n_train = int(len(idc) * train_frac)
        n_eval = len(idc) - n_train
        idc_train, idc_eval = torch.utils.data.random_split(idc, [n_train, n_eval])

        train_X.append(x_train[idc_train])
        train_Y.append(y_train[idc_train])
        samples_user.append(n_train)
        test_X.append(x_train[idc_eval])
        test_Y.append(y_train[idc_eval])
    print("Nodes dataset sizes : ")
    print(samples_user)
    fracs = np.zeros((nodes, nodes), dtype=float)
    for i in range(0, nodes):
        for j in range(0, nodes):
            fracs[i, j] = samples_user[j] / samples_user[i]


                                         ##########################################

    train_Y = [tf.one_hot(train_Y[i], depth=num_classes).numpy() for i in range(0, nodes)]
    test_Y = [tf.one_hot(test_Y[i], depth=num_classes).numpy() for i in range(0, nodes)]
    return (train_X, train_Y), (test_X, test_Y), x_shape, num_classes, fracs, samples_user


def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation),
                                  '')


def dataset_to_numpy(ds):
    """
    Convert tensorflow dataset to numpy arrays
    """
    images = []
    labels = []

    # Iterate over a dataset
    for i, (image, label) in enumerate(tfds.as_numpy(ds)):
        images.append(image)
        labels.append(label)

    return np.squeeze(np.array(images)), np.squeeze(np.array(labels))
