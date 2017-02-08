#!/usr/bin/env python

"""A simple python script template.
"""

from __future__ import print_function
import os
import sys
import argparse
import pickle
import pdb
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
from data import DataLoader
from sklearn.preprocessing import normalize
import random
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)



def get_word2vec_embeddings(word2vec_file, vocab):
    import gensim
    ''' load a pre-trained binary format word2vec into a dictionary
    the model is downloaded from https://docs.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download'''
    vocabset = set(vocab)
    logging.info('total vocab: %d' % len(vocabset))
    logging.info('loading w2v embeddings...')
    word2vec_model = gensim.models.word2vec.Word2Vec.load_word2vec_format(word2vec_file, binary=True)
    word_embeddings = {v.lower():word2vec_model[v] for v in word2vec_model.vocab}
    word2vec_vocab = set(word_embeddings.keys())
    logging.info('total w2v vocab: %d' % len(word2vec_vocab))
    not_in_word2vec = vocabset - word2vec_vocab

    
    for text in not_in_word2vec:
        word_embeddings[text] = np.zeros((1, 300))
        #subregion e.g. 'los angeles, san diego'
        subregions = re.split(',|\s', text)
        count_ = 0
        for subregion in subregions:
            if subregion in word2vec_vocab:
                count_ += 1
                word_embeddings[text] += word_embeddings[subregion]
        if count_ > 1:
            word_embeddings[text] /= count_
            
    w2v_vocab = [v for v in word_embeddings if v in vocabset]
    logging.info('vstacking word vectors into a single matrix...')
    embeddings = np.vstack(tuple([np.asarray(word_embeddings[v]).reshape((1,300)) for v in w2v_vocab])) 
    logging.info('#vocab in the model: %d' %len(w2v_vocab))
    return w2v_vocab, embeddings


def get_vocab(dare_file, vectorizer, freq_words):
    '''
    return all vocab from the geolocation model and dare dataset
    which are not among the top 50k most frequent words.
    Also includes subregions as a single entry (e.g. "los angeles,san diego,san jose").
    It is assumed that localisms we are looking for are not among
    top 50k most frequent English words.
    '''
    dare_vocab = []
    model_vocab = vectorizer.get_feature_names()
    with codecs.open(dare_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            obj = json.loads(line, encoding='utf-8')
            dialect = obj['dialect'].lower()
            subregions = obj['dialect subregions']
            word = obj['word'].lower()
            dare_vocab.append(word)
            if subregions:
                #dare_vocab.append(subregions.lower())
                subregion_items = re.split(',|\s', subregions.lower())
                dare_vocab.extend(subregion_items)
                dare_vocab.append(subregions.lower())
            else:
                dare_vocab.append(dialect)
                dare_vocab.extend(dialect.lower())
    vocab = set(dare_vocab) | set(model_vocab) - freq_words
    vocab = sorted(set([v.strip() for v in vocab if len(v)>1]))
    return vocab
    

def get_frequent_words(word_count_file, topk=50000):
    '''
    read word frequency file from 
    Peter Norvig. 2009. Natural language corpus data. Beautiful Data pages 219-242.
    http://norvig.com/ngrams/count_1w.txt
    and return topk most frequent ones.
    '''
    words = []
    with codecs.open(word_count_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            word, count = line.strip().split('\t')
            words.append(word)
    return set(words[0:topk])

def nearest_neighbours(vocab, embs, k):
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import normalize
    from sklearn.preprocessing import MinMaxScaler

    #now read dare json files
    json_file = './data/geodare.cleansed.filtered.json'
    json_objs = []

    dialect_subregions = {}
    final_dialect_words = defaultdict(set)
    with codecs.open(json_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            obj = json.loads(line, encoding='utf-8')
            json_objs.append(obj)         
            dialect = obj['dialect'].lower()
            subregions = obj['dialect subregions']
            word = obj['word'].lower()
            
            if subregions:
                final_dialect_words[subregions.lower()].add(word)
                dialect_subregions[dialect] = subregions.lower()                
            else:
                final_dialect_words[dialect].add(word)

    
    logging.info('creating dialect embeddings...')
    dialect_embs = OrderedDict()
    vocabset = set(vocab)
    
    covered_dialects = [dialect for dialect in sorted(final_dialect_words) if dialect in vocabset]
    ignored_dialects = [dialect for dialect in final_dialect_words if dialect not in vocabset]
    logging.info('#dare dialects: %d  #dare dialects in the model: %d' %(len(final_dialect_words), len(covered_dialects)))
    logging.info('ignored dialects: %s' % '-'.join(ignored_dialects))
    dialect_indices = [vocab.index(dialect) for dialect in covered_dialects]
    target_X = np.vstack(tuple([embs[i, :].reshape(1, embs.shape[1]) for i in dialect_indices]))

    
    #logging.info('MinMax Scaling each dimension to fit between 0,1')
    #target_X = scaler.fit_transform(target_X)
    #logging.info('l1 normalizing embedding samples')
    #target_X = normalize(target_X, norm='l1', axis=1, copy=False)

    #target_indices = np.asarray(text_index.values())
    #target_X = embs[target_indices, :]
    logging.info('computing nearest neighbours of dialects')
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', leaf_size=10).fit(embs)
    distances, indices = nbrs.kneighbors(target_X)
    word_nbrs = [(covered_dialects[i], vocab[indices[i, j]]) for i in range(target_X.shape[0]) for j in range(k)]
    word_neighbours = defaultdict(list)
    for word_nbr in word_nbrs:
        word, nbr = word_nbr
        word_neighbours[word].append(nbr)
    return word_neighbours

def nearest_neighbours2(vocab, embs, k):
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import normalize
    from sklearn.preprocessing import MinMaxScaler

    #now read dare json files
    json_file = './data/geodare.cleansed.filtered.json'
    json_objs = []

    dialect_subregions = {}
    final_dialect_words = defaultdict(set)
    dialects = set()
    with codecs.open(json_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            obj = json.loads(line, encoding='utf-8')
            json_objs.append(obj)         
            dialect = obj['dialect'].lower()
            subregions = obj['dialect subregions']
            word = obj['word'].lower()
            dialects.add(dialect)
            if subregions:
                final_dialect_words[subregions].add(word)
                dialect_subregions[dialect] = subregions               
            else:
                final_dialect_words[dialect].add(word)

    
    logging.info('creating dialect embeddings...')
    dialect_embs = OrderedDict()
    vocabset = set(vocab)
    dialects_sorted = sorted(dialects)
    for dialect in dialects_sorted:
        subregions = dialect_subregions.get(dialect, None)
        all_dialect_terms = []
        dialect_terms = re.split(',|\s', dialect)
        all_dialect_terms.extend(dialect_terms)
        if subregions:
            subregions_terms = re.split(',|\s', subregions)
            all_dialect_terms.extend(subregions_terms)
        
        dialect_term_indices = [vocab.index(term) for term in all_dialect_terms if term in vocabset]
        dialect_emb = np.ones((1, embs.shape[1]))
        for _index in dialect_term_indices:
            dialect_emb *= embs[_index, :].reshape((1, embs.shape[1]))
        #dialect_emb = dialect_emb / len(dialect_item_indices)
        dialect_embs[dialect] = dialect_emb
    target_X = np.vstack(tuple(dialect_embs.values()))
    
    #logging.info('MinMax Scaling each dimension to fit between 0,1')
    #target_X = scaler.fit_transform(target_X)
    #logging.info('l1 normalizing embedding samples')
    #target_X = normalize(target_X, norm='l1', axis=1, copy=False)

    #target_indices = np.asarray(text_index.values())
    #target_X = embs[target_indices, :]
    logging.info('computing nearest neighbours of dialects')
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', leaf_size=10).fit(embs)
    distances, indices = nbrs.kneighbors(target_X)
    word_nbrs = [(dialects_sorted[i], vocab[indices[i, j]]) for i in range(target_X.shape[0]) for j in range(k)]
    word_neighbours = defaultdict(list)
    for word_nbr in word_nbrs:
        word, nbr = word_nbr
        word_neighbours[word].append(nbr)
    return word_neighbours

def recall_at_k(word_nbrs, k):
    json_file = './data/geodare.cleansed.filtered.json'
    json_objs = []
    texts = []
    dialect_words = defaultdict(list)
    with codecs.open(json_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            obj = json.loads(line, encoding='utf-8')
            json_objs.append(obj)         
            dialect = obj['dialect'].lower()
            subregions = obj['dialect subregions']
            word = obj['word'].lower()
            texts.append(word)
            if subregions:
                dialect_words[subregions].append(word)
            else:
                dialect_words[dialect].append(word)

    recalls = []
    info = []
    total_true_positive = 0
    total_positive = 0
    for dialect, nbrs in word_nbrs.iteritems():
        dialect_has = 0
        dialect_total = 0
        nbrs = set(nbrs[0:k])
        if dialect in dialect_words:
            dwords = set(dialect_words[dialect])
            dialect_total = len(dwords)
            total_positive += dialect_total
            if dialect_total == 0:
                print('zero dialect words ' + dialect)
                continue
            for dword in dwords:
                if dword in nbrs:
                    dialect_has += 1
                    total_true_positive += 1
            recall = 100 * float(dialect_has) / dialect_total
            recalls.append(recall)
            info.append((dialect, dialect_total, recall))
        else:
            print('this dialect does not exist: ' + dialect)
    print('recall at ' + str(k))
    #print(len(recalls))
    #print(np.mean(recalls))
    #print(np.median(recalls))
    #print(info)
    sum_support = sum([inf[1] for inf in info])
    #weighted_average_recall = sum([inf[1] * inf[2] for inf in info]) / sum_support
    #print('weighted average recall: ' + str(weighted_average_recall))
    print('micro recall :' + str(float(total_true_positive) * 100 / total_positive))

def recall_at_k2(word_nbrs, k):
    json_file = './data/geodare.cleansed.filtered.json'
    json_objs = []
    texts = []
    dialect_words = defaultdict(list)
    with codecs.open(json_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            obj = json.loads(line, encoding='utf-8')
            json_objs.append(obj)         
            dialect = obj['dialect'].lower()
            subregions = obj['dialect subregions']
            word = obj['word'].lower()
            texts.append(word)
            dialect_words[dialect].append(word)

    recalls = []
    info = []
    total_true_positive = 0
    total_positive = 0
    for dialect, nbrs in word_nbrs.iteritems():
        dialect_has = 0
        dialect_total = 0
        nbrs = set(nbrs[0:k])
        if dialect in dialect_words:
            dwords = set(dialect_words[dialect])
            dialect_total = len(dwords)
            total_positive += dialect_total
            if dialect_total == 0:
                print('zero dialect words ' + dialect)
                continue
            for dword in dwords:
                if dword in nbrs:
                    dialect_has += 1
                    total_true_positive += 1
            recall = 100 * float(dialect_has) / dialect_total
            recalls.append(recall)
            info.append((dialect, dialect_total, recall))
        else:
            print('this dialect does not exist: ' + dialect)
    print('recall at ' + str(k))
    #print(len(recalls))
    #print(np.mean(recalls))
    #print(np.median(recalls))
    #print(info)
    sum_support = sum([inf[1] for inf in info])
    #weighted_average_recall = sum([inf[1] * inf[2] for inf in info]) / sum_support
    #print('weighted average recall: ' + str(weighted_average_recall))
    print('micro recall :' + str(float(total_true_positive) * 100 / total_positive))

def dialect_eval(embs_file='./word-embs-10000-1e-06-1e-06.pkl.gz', word2vec=None, lr=None):
    logging.info('word2vec: ' + str(word2vec) + " lr: " + str(lr))
    logging.info('loading vocab, embs from ' + embs_file)
    with gzip.open(embs_file, 'rb') as fin:
        vocab, embs = pickle.load(fin)
    vocab_size = len(vocab)
    print('vocab size: ' + str(vocab_size))
    if word2vec:
        vocabset = set(vocab)
        logging.info('loading w2v embeddings...')
        word2vec_model = load_word2vec('/home/arahimi/GoogleNews-vectors-negative300.bin.gz')
        w2v_vocab = [v for v in word2vec_model.vocab if v in vocabset]
        logging.info('vstacking word vectors into a single matrix...')
        w2v_embs = np.vstack(tuple([np.asarray(word2vec_model[v]).reshape((1,300)) for v in w2v_vocab]))
        embs = w2v_embs
        vocab = w2v_vocab 
    elif lr:
        with open('/home/arahimi/datasets/na-original/model-na-original-median-2400-1e-06.pkl', 'rb') as fout:
            clf, vectorizer = pickle.load(fout)
        X_lr = vectorizer.transform(vocab)
        lr_embs = clf.predict_proba(X_lr) 
        embs = lr_embs       

    word_nbrs = nearest_neighbours(vocab, embs, k=int(len(vocab)))
    
    percents = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
    percents = [int(p* vocab_size) for p in percents]
    #percents = [10000, 20000, 30000, 40000, 50000]
    for r_at_k in percents:
        recall_at_k(word_nbrs=word_nbrs, k=r_at_k)

def eval_embeddings(vocab, embeddings):
    '''
    Given a embeddings and the corresponding vocab
    finds the nearest neighbours of each vocab and then
    given a dialect region/subregion tries to find localisms
    within the nearest neighbours and reports recall at k.
    '''
    from sklearn.preprocessing import normalize
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1), copy=False)
    logging.info('MinMax Scaling each dimension to fit between 0,1')
    embeddings = scaler.fit_transform(embeddings)
    logging.info('l1 normalizing embedding samples')
    embeddings = normalize(embeddings, norm='l1', axis=1, copy=False)
    word_nbrs = nearest_neighbours(vocab, embeddings, k=int(len(vocab)))
    #percents = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
    vocab_size = len(vocab)
    #percents = [int(p* vocab_size) for p in percents]
    percents = [10000, 20000, 30000, 40000, 50000]
    for r_at_k in percents:
        recall_at_k(word_nbrs=word_nbrs, k=r_at_k)

def get_lr_embeddings(vocab):
    '''
    loads a trained logistic regression model, predicts location
    probabilities and uses that as embeddings for each word/dialect region/subregion.
    '''
    with open('./data/model-na-original-median-2400-1e-06.pkl', 'rb') as fout:
        clf, vectorizer = pickle.load(fout)
    X_lr = vectorizer.transform(vocab)
    lr_embs = clf.predict_proba(X_lr)
    return vocab, lr_embs 

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

def get_mlp_embeddings(**kwargs):
    data = kwargs.get('data')
    vocab = kwargs.get('vocab')
    clf = MLP(n_epochs=50, batch_size=10000, init_parameters=None, complete_prob=False, 
              add_hidden=True, regul_coefs=[1e-6, 1e-6], save_results=False, hidden_layer_size=2048, 
              drop_out=True, drop_out_coefs=[0.5, 0.5], early_stopping_max_down=5, loss_name='log', nonlinearity='rectify')
    metainfo, X_train, Y_train, U_train, X_dev, Y_dev, U_dev, X_test, Y_test, U_test, classLatMedian, classLonMedian, userLocation, vectorizer = data
    convolution = False
    if convolution:
        logging.info('loading graph...')
        with open('/home/arahimi/git/jointgeo/data/trans.cmu.graph', 'rb') as fin:
            dev_graph = pickle.load(fin)
        '''
        dev_graph_indices = xrange(X_train.shape[0], X_train.shape[0] + X_dev.shape[0])
        X_test = X_test.tolil()
        for i in dev_graph_indices:
            nbrs = dev_graph[i]
            dev_index = i - X_train.shape[0]
            count = 1
            for nbr in nbrs:
                if nbr < X_train.shape[0]:
                    X_test[i - X_train.shape[0], :] += X_train[nbr, :]
                    count += 1
            X_test[i - X_train.shape[0], :] /= count
        X_test = X_test.tocsr().astype('float32')
        '''
        for i in range(0, X_train.shape[0] + X_dev.shape[0]):
            dev_graph[i].append(i)
        logging.info('creating adjacency matrix...')
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(dev_graph))
        adj.setdiag(1)
        pdb.set_trace()
        logging.info('normalizing adjacency matrix...')
        normalize(adj, axis=1, norm='l1', copy=False)
        adj = adj.astype('float32')
        logging.info('vstacking...')
        X = sp.sparse.vstack([X_train, X_test])
        logging.info('convolution...')
        X_conv = adj * X
        X_conv = X_conv.tocsr().astype('float32')
        #X_train = X_conv[0:X_train.shape[0], :]
        X_test = X_conv[X_train.shape[0]:, :]
    
    clf.fit(X_train, Y_train, X_dev, Y_dev)
    print('Test classification accuracy is %f' % clf.accuracy(X_test, Y_test))
    y_pred = clf.predict(X_test)
    geo_eval(Y_test, y_pred, U_test, classLatMedian, classLonMedian, userLocation)
    print('Dev classification accuracy is %f' % clf.accuracy(X_dev, Y_dev))
    y_pred = clf.predict(X_dev)
    geo_eval(Y_dev, y_pred, U_dev, classLatMedian, classLonMedian, userLocation)
    
    X_dare = vectorizer.transform(vocab)
    X_dare = X_dare.astype('float32')
    mlp_embeddings = clf.get_embedding(X_dare)
    return vocab, mlp_embeddings

    

def main(**kwargs):
    args = parse_args(sys.argv[1:])
    data = load_data(dataset=args.dataset)
    #get_mlp_embeddings(data=data)
    freq_words = get_frequent_words(word_count_file='./data/count_1w.txt', topk=50000)
    vocab = get_vocab(dare_file='./data/geodare.cleansed.filtered.json', vectorizer=data[-1], freq_words=freq_words)
    vocab_lr, embeddings = get_lr_embeddings(vocab)
    eval_embeddings(vocab_lr, embeddings)
    #vocab_w2v, embeddings = get_word2vec_embeddings(word2vec_file='./data/GoogleNews-vectors-negative300.bin.gz', vocab=vocab)
    #eval_embeddings(vocab_w2v, embeddings)
    #vocab_mlp, embeddings = get_mlp_embeddings(data=data, vocab=vocab)
    #eval_embeddings(vocab_mlp, embeddings)
     
def main2(data_home, **kwargs):
    bucket_size = kwargs.get('bucket', 300)
    batch_size = kwargs.get('batch', 500)
    hidden_size = kwargs.get('hidden', 500)
    encoding = kwargs.get('encoding', 'utf-8')
    regul = kwargs.get('regularization', 1e-6)
    celebrity_threshold = kwargs.get('celebrity', 10)
    convolution = kwargs.get('conv', False)
    
    dl = DataLoader(data_home=data_home, bucket_size=bucket_size, encoding=encoding, celebrity_threshold=celebrity_threshold)
    dl.load_data()
    dl.get_graph()
    dl.assignClasses()
    dl.tfidf()
    
    U_test = dl.df_test.index.tolist()
    U_dev = dl.df_dev.index.tolist()
    U_train = dl.df_train.index.tolist()  
    if convolution:  
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
          drop_out=True, drop_out_coefs=[0.5, 0.5], early_stopping_max_down=10, loss_name='log', nonlinearity='rectify')
    clf.fit(X_train, Y_train, X_dev, Y_dev)
    print('Test classification accuracy is %f' % clf.accuracy(X_test, Y_test))
    y_pred = clf.predict(X_test)
    geo_eval(Y_test, y_pred, U_test, classLatMedian, classLonMedian, userLocation)
    print('Dev classification accuracy is %f' % clf.accuracy(X_dev, Y_dev))
    y_pred = clf.predict(X_dev)
    mean, median, acc161 = geo_eval(Y_dev, y_pred, U_dev, classLatMedian, classLonMedian, userLocation)
    return mean, median , acc161

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
    args = parser.parse_args(argv)
    return args
def tune(data_home):
    celeb = 5 if 'cmu' in data_home else 10
    bucket = 50 if 'cmu' in data_home else 2400
    encoding = 'latin1' if 'cmu' in data_home else 'utf-8'
    results = []
    for i in range(50):
        batch = 200 if 'cmu' in data_home else 5000
        hidden = random.choice([200, 800, 3200]) if 'cmu' in data_home else random.choice([800, 3200, 6400])
        regularization = random.choice([ 1e-5, 5e-6, 1e-6, 5e-7, 1e-7])
        print('iter %d, batch %d, hidden %d, regul %f' %(i, batch, hidden, regularization))
        mean, median, acc161 = main2(data_home=data_home, batch=batch, hidden=hidden, 
              encoding=encoding, regularization=regularization,
              celebrity_threshold=celeb, bucket=bucket)
        results.append((celeb, batch, hidden, regularization, mean, median, acc161))
    for result in results:
        print(result)
        
    
if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    #nice -n 18 python main.py  -hid 500 -bucket 2400 -batch 10000 -d ~/datasets/na/processed_data/ -enc utf-8 -reg 1e-6 -cel 15
    
    main2(data_home=args.dir, batch=args.batch, hidden=args.hidden, 
          encoding=args.encoding, regularization=args.regularization,
          celebrity_threshold=args.celebrity, bucket=args.bucket)
    '''
    #tune(data_home=args.dir)
    main()
    '''