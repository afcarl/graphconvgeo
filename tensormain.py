#!/usr/bin/env python

"""A simple python script template.
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import sys
import argparse
import pickle
import pdb
from mlpconv import MLPCONV
from mlp import MLP
import logging
import numpy as np
from haversine import haversine
import gzip
import codecs
from collections import OrderedDict, defaultdict
import json
import re
import networkx as nx
import scipy as sp
from sklearn.preprocessing import normalize, OneHotEncoder
from data import DataLoader
import random

import argparse
import sys 
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

np.random.seed(77)


def geo_eval(y_true, y_pred, U_eval, classLatMedian, classLonMedian, userLocation):
    assert len(y_pred) == len(U_eval), "#preds: %d, #users: %d" %(len(y_pred), len(U_eval))
    distances = []
    for i in range(0, len(y_pred)):
        user = U_eval[i]
        location = userLocation[user].split(',')
        lat, lon = float(location[0]), float(location[1])
        prediction = str(y_pred[i])
        lat_pred, lon_pred = classLatMedian[prediction], classLonMedian[prediction]  
        distance = haversine((lat, lon), (lat_pred, lon_pred))
        distances.append(distance)

    acc_at_161 = 100 * len([d for d in distances if d < 161]) / float(len(distances))

    logging.info( "Mean: " + str(int(np.mean(distances))) + " Median: " + str(int(np.median(distances))) + " Acc@161: " + str(int(acc_at_161)))
        
    return np.mean(distances), np.median(distances), acc_at_161

def load_data(**kwargs):
    logging.info('loading data...')
    with open(os.path.join('./data', kwargs.get('dataset')) + '.pkl', 'rb') as fin:
        data = pickle.load(fin)
    #data[0] = 'X_train, Y_train, U_train, X_dev, Y_dev, U_dev, X_test, Y_test, U_test, classLatMedian, classLonMedian, userLocation, vectorizer'
    return data



def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]    


     
def main_justinputconv(data_home, **kwargs):
    bucket_size = kwargs.get('bucket', 300)
    batch_size = kwargs.get('batch', 500)
    hidden_size = kwargs.get('hidden', 500)
    encoding = kwargs.get('encoding', 'utf-8')
    regul = kwargs.get('regularization', 1e-6)
    celebrity_threshold = kwargs.get('celebrity', 10)
    convolution = kwargs.get('convolution', False)    
    dl = DataLoader(data_home=data_home, bucket_size=bucket_size, encoding=encoding, celebrity_threshold=celebrity_threshold)
    dl.load_data()
    dl.assignClasses()
    dl.tfidf()
    U_test = dl.df_test.index.tolist()
    U_dev = dl.df_dev.index.tolist()
    U_train = dl.df_train.index.tolist()    
    if convolution:
        dl.get_graph()  
        logging.info('creating adjacency matrix...')
        adj = nx.adjacency_matrix(dl.graph, nodelist=xrange(len(U_train + U_dev + U_test)), weight='w')
        #adj[adj > 0] = 1
        adj.setdiag(1)
        n,m = adj.shape
        diags = adj.sum(axis=1).flatten()
        with sp.errstate(divide='ignore'):
            diags_sqrt = 1.0/sp.sqrt(diags)
        diags_sqrt[sp.isinf(diags_sqrt)] = 0
        D_pow_neghalf = sp.sparse.spdiags(diags_sqrt, [0], m, n, format='csr')
        H = D_pow_neghalf * adj * D_pow_neghalf
        
        #logging.info('normalizing adjacency matrix...')
        #normalize(adj, axis=1, norm='l1', copy=False)
        #adj = adj.astype('float32')
        logging.info('vstacking...')
        X = sp.sparse.vstack([dl.X_train, dl.X_dev, dl.X_test])
        logging.info('convolution...')
        X_conv = H * X
        X_conv = X_conv.tocsr().astype('float32')
        X_train = X_conv[0:dl.X_train.shape[0], :]
        X_dev = X_conv[dl.X_train.shape[0]:dl.X_train.shape[0] + dl.X_dev.shape[0], :]
        X_test = X_conv[dl.X_train.shape[0] + dl.X_dev.shape[0]:, :]
    else:
        X_train = dl.X_train
        X_dev = dl.X_dev
        X_test = dl.X_test
    Y_test = dl.test_classes
    Y_train = dl.train_classes
    Y_dev = dl.dev_classes
    classLatMedian = {str(c):dl.cluster_median[c][0] for c in dl.cluster_median}
    classLonMedian = {str(c):dl.cluster_median[c][1] for c in dl.cluster_median}


    P_test = [str(a[0]) + ',' + str(a[1]) for a in dl.df_test[['lat', 'lon']].values.tolist()]
    P_train = [str(a[0]) + ',' + str(a[1]) for a in dl.df_train[['lat', 'lon']].values.tolist()]
    P_dev = [str(a[0]) + ',' + str(a[1]) for a in dl.df_dev[['lat', 'lon']].values.tolist()]
    userLocation = {}
    for i, u in enumerate(U_train):
        userLocation[u] = P_train[i]
    for i, u in enumerate(U_test):
        userLocation[u] = P_test[i]
    for i, u in enumerate(U_dev):
        userLocation[u] = P_dev[i]
    
    clf = MLP(n_epochs=200, batch_size=batch_size, init_parameters=None, complete_prob=False, 
          add_hidden=True, regul_coefs=[regul, regul], save_results=False, hidden_layer_size=hidden_size, 
          drop_out=True, drop_out_coefs=[0.5, 0.5], early_stopping_max_down=5, loss_name='log', nonlinearity='rectify')
    clf.fit(X_train, Y_train, X_dev, Y_dev)
    print('Test classification accuracy is %f' % clf.accuracy(X_test, Y_test))
    y_pred = clf.predict(X_test)
    geo_eval(Y_test, y_pred, U_test, classLatMedian, classLonMedian, userLocation)
    print('Dev classification accuracy is %f' % clf.accuracy(X_dev, Y_dev))
    y_pred = clf.predict(X_dev)
    geo_eval(Y_dev, y_pred, U_dev, classLatMedian, classLonMedian, userLocation)

def preprocess_data(data_home, **kwargs):
    bucket_size = kwargs.get('bucket', 300)
    encoding = kwargs.get('encoding', 'utf-8')
    celebrity_threshold = kwargs.get('celebrity', 10)  
    mindf = kwargs.get('mindf', 10)
    dl = DataLoader(data_home=data_home, bucket_size=bucket_size, encoding=encoding, 
                    celebrity_threshold=celebrity_threshold, one_hot_labels=False, mindf=mindf)
    dl.load_data()
    dl.assignClasses()
    dl.tfidf()
    U_test = dl.df_test.index.tolist()
    U_dev = dl.df_dev.index.tolist()
    U_train = dl.df_train.index.tolist()    

    dl.get_graph()  
    logging.info('creating adjacency matrix...')
    adj = nx.adjacency_matrix(dl.graph, nodelist=xrange(len(U_train + U_dev + U_test)), weight='w')
    #adj[adj > 0] = 1
    adj.setdiag(1)
    n,m = adj.shape
    diags = adj.sum(axis=1).flatten()
    with sp.errstate(divide='ignore'):
        diags_sqrt = 1.0/sp.sqrt(diags)
    diags_sqrt[sp.isinf(diags_sqrt)] = 0
    D_pow_neghalf = sp.sparse.spdiags(diags_sqrt, [0], m, n, format='csr')
    H = D_pow_neghalf * adj * D_pow_neghalf
    H = H.astype('float32')
    logging.info('adjacency matrix created.')
    
    X_train = dl.X_train
    X_dev = dl.X_dev
    X_test = dl.X_test
    Y_test = dl.test_classes
    Y_train = dl.train_classes
    Y_dev = dl.dev_classes
    classLatMedian = {str(c):dl.cluster_median[c][0] for c in dl.cluster_median}
    classLonMedian = {str(c):dl.cluster_median[c][1] for c in dl.cluster_median}
    
    
    P_test = [str(a[0]) + ',' + str(a[1]) for a in dl.df_test[['lat', 'lon']].values.tolist()]
    P_train = [str(a[0]) + ',' + str(a[1]) for a in dl.df_train[['lat', 'lon']].values.tolist()]
    P_dev = [str(a[0]) + ',' + str(a[1]) for a in dl.df_dev[['lat', 'lon']].values.tolist()]
    userLocation = {}
    for i, u in enumerate(U_train):
        userLocation[u] = P_train[i]
    for i, u in enumerate(U_test):
        userLocation[u] = P_test[i]
    for i, u in enumerate(U_dev):
        userLocation[u] = P_dev[i]
    
    data = (H, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test, classLatMedian, classLonMedian, userLocation)
    return data

def main_mlpconv(data, **kwargs):
    batch_size = kwargs.get('batch', 500)
    hidden_size = kwargs.get('hidden', 500)
    dropout_coefs = kwargs.get('dropout_coefs', [0.5, 0.5])
    regul = kwargs.get('regularization', 1e-6)
    H, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test, classLatMedian, classLonMedian, userLocation = data
    logging.info('stacking training, dev and test features and creating indices...')
    X = sp.sparse.vstack([X_train, X_dev, X_test])
    X = X.astype('float32')
    train_indices = np.asarray(range(0, X_train.shape[0])).astype('int32')
    dev_indices = np.asarray(range(X_train.shape[0], X_train.shape[0] + X_dev.shape[0])).astype('int32')
    test_indices = np.asarray(range(X_train.shape[0] + X_dev.shape[0], X_train.shape[0] + X_dev.shape[0] + X_test.shape[0])).astype('int32')
    logging.info('running mlp with graph conv...')
    clf = MLPCONV(n_epochs=500, batch_size=batch_size, init_parameters=None, complete_prob=False, 
          add_hidden=True, regul_coefs=[regul, regul], save_results=False, hidden_layer_size=hidden_size, 
          drop_out=False, dropout_coefs=dropout_coefs, early_stopping_max_down=10, loss_name='log', nonlinearity='rectify')
    clf.fit(X, train_indices, dev_indices, test_indices, Y_train, Y_dev, Y_test, H)
    print('Test classification accuracy is %f' % clf.accuracy(dataset_partition='test', y_true=Y_test.astype('int32')))
    y_pred = clf.predict(dataset_partition='test')
    geo_eval(Y_test, y_pred, U_test, classLatMedian, classLonMedian, userLocation)
    print('Dev classification accuracy is %f' % clf.accuracy(dataset_partition='dev', y_true=Y_dev.astype('int32')))
    y_pred = clf.predict(dataset_partition='dev')
    geo_eval(Y_dev, y_pred, U_dev, classLatMedian, classLonMedian, userLocation)

def tune(args):
    data = preprocess_data(data_home=args.dir, encoding=args.encoding, celebrity=args.celebrity, bucket=args.bucket, mindf=args.mindf)
    for i in range(50):
        random.seed()
        if '/cmu/' in args.dir:
            hidden_layer_size = random.choice([32 * x for x in range(2, 50)])
            batch_size = 100
            encoding = 'latin1'
        elif '/na/' in args.dir:
            hidden_layer_size = random.choice([256 * x for x in range(2, 10)])
            batch_size = 10000
            encoding = 'utf-8'
        elif '/world' in args.dir:
            hidden_layer_size = random.choice([930 * x for x in range(2, 10)])
            batch_size = 10000
            encoding = 'utf-8' 
        regul = random.choice([1e-6, 1e-7])
        dropout_coefs = random.choice([[0, x] for x in [0.4, 0.5, 0.6] ])
        np.random.seed(77) 
        logging.info('#iter %d, regul %s, hidden %d, drop %s' %(i, str(regul), hidden_layer_size, str(dropout_coefs)))
        main_mlpconv(data, batch=batch_size, hidden=hidden_layer_size, 
                     regularization=regul, dropout_coefs=dropout_coefs)
        

    
def parse_args(argv):
    """
    Parse commandline arguments.
    Arguments:
        argv -- An argument list without the program name.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i','--dataset', metavar='str',
        help='dataset for dialectology',
        type=str, default='na')
    parser.add_argument(
        '-bucket','--bucket', metavar='int',
        help='discretisation bucket size',
        type=int, default=300)
    parser.add_argument(
        '-batch','--batch', metavar='int',
        help='SGD batch size',
        type=int, default=500)
    parser.add_argument(
        '-hid','--hidden', metavar='int',
        help='Hidden layer size',
        type=int, default=500)
    parser.add_argument(
        '-mindf','--mindf', metavar='int',
        help='minimum document frequency in BoW',
        type=int, default=10)
    parser.add_argument(
        '-d','--dir', metavar='str',
        help='home directory',
        type=str, default='./data')
    parser.add_argument(
        '-enc','--encoding', metavar='str',
        help='Data Encoding (e.g. latin1, utf-8)',
        type=str, default='utf-8')
    parser.add_argument(
        '-reg','--regularization', metavar='float',
        help='regularization coefficient)',
        type=float, default=1e-6)
    parser.add_argument(
        '-cel','--celebrity', metavar='int',
        help='celebrity threshold',
        type=int, default=10)
    
    parser.add_argument(
        '-conv', '--convolution', action='store_true',
        help='if true do convolution')
    parser.add_argument(
        '-tune', '--tune', action='store_true',
        help='if true tune the hyper-parameters')   
    args = parser.parse_args(argv)
    return args


if __name__ == '__main__':
    #nohup nice -n 9 python -u tensormain.py -hid 800 -bucket 2400 -batch 10000 -d ~/datasets/na/processed_data/ -enc utf-8 -reg 1e-6 -cel 15 -tune &>> na_conv.txt
    args = parse_args(sys.argv[1:])
    if args.tune:
        tune(args)
    else:
        data = preprocess_data(data_home=args.dir, encoding=args.encoding, celebrity=args.celebrity, bucket=args.bucket, mindf=args.mindf)
        main_mlpconv(data, batch=args.batch, hidden=args.hidden, 
                     regularization=args.regularization)
    
