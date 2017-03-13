'''
Created on 21 Feb 2017
Given a training set of lat/lon as input and probability distribution over words as output,
train a model that can predict words based on location.
then try to visualise borders and regions (e.g. try many lat/lon as input and get the probability of word yinz
in the output and visualise that).
@author: af
'''
import matplotlib as mpl
import re
import lasagne_layers
mpl.use('Agg')
from matplotlib import ticker
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
from matplotlib.patches import Polygon as MplPolygon
import seaborn as sns
sns.set(style="white")
import operator
import argparse
import sys
from scipy.spatial import ConvexHull
import os
import pdb
import random
from data import DataLoader
import numpy as np
import sys
from os import path
import scipy as sp
import theano
import shutil
import theano.tensor as T
import lasagne
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
from lasagne.regularization import regularize_layer_params
import theano.sparse as S
from lasagne.layers import DenseLayer, DropoutLayer
import logging
import json
import codecs
import pickle
import gzip
from collections import OrderedDict, Counter
from sklearn.preprocessing import normalize
from haversine import haversine
from _collections import defaultdict
from scipy import stats
from twokenize import tokenize
from mpl_toolkits.basemap import Basemap, cm, maskoceans
from scipy.interpolate import griddata as gd
from lasagne_layers import SparseInputDenseLayer, GaussianRBFLayer, DiagonalBivariateGaussianLayer, BivariateGaussianLayer
from shapely.geometry import MultiPoint, Point, Polygon
import shapefile
from utils import short_state_names, stop_words, get_us_city_name
from sklearn.cluster import KMeans, MiniBatchKMeans
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

np.random.seed(77)
random.seed(77)

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

class NNModel():
    def __init__(self, 
                 n_epochs=10, 
                 batch_size=1000, 
                 regul_coef=1e-6,
                 input_size=None,
                 output_size = None, 
                 hidden_layer_sizes=None, 
                 drop_out=False, 
                 dropout_coef=0.5,
                 early_stopping_max_down=10,
                 dtype='float32',
                 autoencoder=False,
                 input_sparse=False,
                 reload=False,
                 rbf=False,
                 bigaus=False,
                 mus=None):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.regul_coef = regul_coef
        self.hidden_layer_sizes = hidden_layer_sizes
        self.drop_out = drop_out
        self.dropout_coef = dropout_coef
        self.early_stopping_max_down = early_stopping_max_down
        self.dtype = dtype
        self.input_size = input_size
        self.output_size = output_size
        self.autoencoder = autoencoder
        self.sparse = input_sparse
        self.reload = reload
        self.rbf = rbf
        self.bigaus = bigaus
        self.mus = mus
        logging.info('building nn model with hidden size %s and output size %d input_sparse %s' % (str(self.hidden_layer_sizes), self.output_size, str(self.sparse)))
        self.build()
        
        
    def build(self):

        self.X_sym = S.csr_matrix(name='inputs', dtype=self.dtype)
        self.Y_sym = T.ivector()
        if self.autoencoder:
            logging.info('autoencoder is on!')
        else:
            logging.info('autoencoder is off!')
        l_in = lasagne.layers.InputLayer(shape=(None, self.input_size),
                                         input_var=self.X_sym)

        if self.drop_out and self.dropout_coef > 0:
            l_in = lasagne_layers.SparseInputDenseLayer(l_in, p=self.dropout_coef)

        l_hid1 = SparseInputDenseLayer(l_in, num_units=self.hidden_layer_sizes[0], 
                                      nonlinearity=lasagne.nonlinearities.rectify,
                                      W=lasagne.init.GlorotUniform())
        if self.drop_out and self.dropout_coef > 0:
            l_hid1 = lasagne.layers.dropout(l_hid1, p=self.dropout_coef)

        if self.bigaus:
            logging.info('adding bivariate gaussian layer...')
            l_hid2 = BivariateGaussianLayer(l_hid1, num_units=self.hidden_layer_sizes[1], mus=self.mus)
        else:
            l_hid2 = lasagne.layers.DenseLayer(l_hid1, num_units=self.hidden_layer_sizes[1], 
                                               nonlinearity=lasagne.nonlinearities.rectify,
                                           W=lasagne.init.GlorotUniform())

        l_hid3 = lasagne.layers.DenseLayer(l_hid2, num_units=self.hidden_layer_sizes[2], 
                                           nonlinearity=lasagne.nonlinearities.linear,
                                           W=lasagne.init.GlorotUniform())
 
        self.l_out = lasagne.layers.DenseLayer(l_hid3, num_units=self.output_size,
                                          nonlinearity=lasagne.nonlinearities.softmax,
                                          W=lasagne.init.GlorotUniform())
        
        
        self.eval_output = lasagne.layers.get_output(self.l_out, self.X_sym, deterministic=True)
        self.eval_pred = self.eval_output.argmax(-1)

        #self.mu_output = lasagne.layers.get_output(l_hid, self.X_sym)
        #self.f_mu = theano.function([self.X_sym, self.Y_sym], self.mu_output, on_unused_input='warn')
        #self.embedding = lasagne.lasagne_layers.get_output(l_hid1, self.X_sym, H=H,  deterministic=True)        
        #self.f_get_embeddings = theano.function([self.X_sym], self.embedding)
        self.output = lasagne.layers.get_output(self.l_out, self.X_sym)
        #self.debug_output = lasagne.layers.get_output(l_hid, self.X_sym)
        #self.pred = self.output.argmax(-1)
        loss = lasagne.objectives.categorical_crossentropy(self.output, self.Y_sym)
        loss = loss.mean()
        eval_loss = lasagne.objectives.categorical_crossentropy(self.eval_output, self.Y_sym)
        eval_loss = eval_loss.mean()
        eval_cross_entropy_loss = lasagne.objectives.categorical_crossentropy(self.eval_output, self.Y_sym)
        eval_cross_entropy_loss = eval_cross_entropy_loss.mean()
        
        
        
        if self.regul_coef:
            l1_share_out = 0.0
            l1_share_hid = 0.5
            regul_coef_out, regul_coef_hid = self.regul_coef, self.regul_coef
            logging.info('regul coefficient for output and hidden lasagne_layers is ' + str(self.regul_coef))
            l1_penalty = lasagne.regularization.regularize_layer_params(self.l_out, l1) * regul_coef_out * l1_share_out
            l2_penalty = lasagne.regularization.regularize_layer_params(self.l_out, l2) * regul_coef_out * (1-l1_share_out)
            l1_penalty += lasagne.regularization.regularize_layer_params(l_hid1, l1) * regul_coef_hid * l1_share_hid
            l2_penalty += lasagne.regularization.regularize_layer_params(l_hid1, l2) * regul_coef_hid * (1-l1_share_hid)
            l1_penalty += lasagne.regularization.regularize_layer_params(l_hid3, l1) * regul_coef_hid * l1_share_hid
            l2_penalty += lasagne.regularization.regularize_layer_params(l_hid3, l2) * regul_coef_hid * (1-l1_share_hid)

            loss = loss + l1_penalty + l2_penalty
            eval_loss = eval_loss + l1_penalty + l2_penalty

        
        parameters = lasagne.layers.get_all_params(self.l_out, trainable=True)
        updates = lasagne.updates.adamax(loss, parameters, learning_rate=2e-3, beta1=0.9, beta2=0.999, epsilon=1e-8)
        self.f_train = theano.function([self.X_sym, self.Y_sym], loss, updates=updates, on_unused_input='warn')
        self.f_val = theano.function([self.X_sym, self.Y_sym], eval_loss, on_unused_input='warn')
        self.f_cross_entropy_loss = theano.function([self.X_sym, self.Y_sym], eval_cross_entropy_loss, on_unused_input='warn')
        self.f_predict_proba = theano.function([self.X_sym], self.eval_output, on_unused_input='warn') 
        self.f_predict = theano.function([self.X_sym], self.eval_pred, on_unused_input='warn')
        #self.f_debug_output =  theano.function([self.X_sym], self.debug_output, on_unused_input='warn')

    def set_params(self, params):
        lasagne.layers.set_all_param_values(self.l_out, params)
    def iterate_minibatches(self, inputs, targets, batchsize, shuffle=False):
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
    
    def fit(self, X_train, Y_train, X_dev, Y_dev, X_test, Y_test):
        model_file = './data/lang2loc_model_' + str(self.output_size) + '_' + str(self.hidden_layer_sizes)  + '_encoder_' + \
        str(self.autoencoder) + '_sparse_' + str(self.sparse) +  '_rbf_' + str(self.rbf) + '_bigaus_' + str(self.bigaus) + '.pkl'
        if self.reload:
            if path.exists(model_file):
                logging.info('loading the model from %s' %model_file)
                with open(model_file, 'rb') as fin:
                    params = pickle.load(fin)
                self.set_params(params)
                    
        logging.info('training with %d n_epochs and  %d batch_size' %(self.n_epochs, self.batch_size))
        best_params = None
        best_val_loss = sys.maxint
        n_validation_down = 0
        
        for step in range(self.n_epochs):
            for batch in self.iterate_minibatches(X_train, Y_train, self.batch_size, shuffle=True):
                x_batch, y_batch = batch
                l_train = self.f_train(x_batch, y_batch)
            l_val = self.f_cross_entropy_loss(X_dev, Y_dev)
            if l_val < best_val_loss and (best_val_loss - l_val) > (0.0001 * l_val):
                best_val_loss = l_val
                best_params = lasagne.layers.get_all_param_values(self.l_out)
                n_validation_down = 0
            else:
                n_validation_down += 1
                if n_validation_down > self.early_stopping_max_down:
                    logging.info('validation results went down. early stopping ...')
                    break
            logging.info('iter %d, train loss %f, dev loss %f, best dev loss %f, num_down %d' %(step, l_train, l_val, best_val_loss, n_validation_down))
        lasagne.layers.set_all_param_values(self.l_out, best_params)
        logging.info('dumping the model...')
        with open(model_file, 'wb') as fout:
            pickle.dump(best_params, fout)
                
    def predict(self, X):
        pred = self.f_predict(X)
        return pred       


          


def get_cluster_centers(input, n_cluster):
    #kmns = KMeans(n_clusters=n_cluster, n_jobs=10)
    kmns = MiniBatchKMeans(n_clusters=n_cluster, batch_size=1000)
    kmns.fit(input)
    return kmns.cluster_centers_.astype('float32')
    
          
def load_data(data_home, **kwargs):
    bucket_size = kwargs.get('bucket', 300)
    dataset_name = kwargs.get('dataset_name')
    encoding = kwargs.get('encoding', 'utf-8')
    celebrity_threshold = kwargs.get('celebrity', 10)  
    mindf = kwargs.get('mindf', 10)
    dtype = kwargs.get('dtype', 'float32')
    one_hot_label = kwargs.get('onehot', False)
    grid_transform = kwargs.get('grid', False)
    normalize_words = kwargs.get('norm', False)
    city_stops = kwargs.get('city_stops', False)

    dl = DataLoader(data_home=data_home, bucket_size=bucket_size, encoding=encoding, 
                    celebrity_threshold=celebrity_threshold, one_hot_labels=one_hot_label, 
                    mindf=mindf, maxdf=0.1, norm='l2', idf=True, btf=True, tokenizer=None, subtf=True, stops=stop_words)
    dl.load_data()
    dl.assignClasses()
    dl.tfidf()
    U_test = dl.df_test.index.tolist()
    U_dev = dl.df_dev.index.tolist()
    U_train = dl.df_train.index.tolist()  
    X_train = dl.X_train.astype('float32')
    X_dev = dl.X_dev.astype('float32')
    X_test = dl.X_test.astype('float32')
    Y_test = dl.test_classes.astype('int32')
    Y_train = dl.train_classes.astype('int32')
    Y_dev = dl.dev_classes.astype('int32')
    classLatMedian = {str(c):dl.cluster_median[c][0] for c in dl.cluster_median}
    classLonMedian = {str(c):dl.cluster_median[c][1] for c in dl.cluster_median}
    loc_train = np.array([[a[0], a[1]] for a in dl.df_train[['lat', 'lon']].values.tolist()], dtype='float32')
    
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
    
    data = (X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test, classLatMedian, classLonMedian, userLocation, loc_train)
    return data
    
def train(data, **kwargs):
    dropout_coef = kwargs.get('dropout_coef', 0.5)
    regul = kwargs.get('regul_coef', 1e-6)
    hid_size = kwargs.get('hidden_size', 200)
    autoencoder = kwargs.get('autoencoder', False)
    grid_transform = kwargs.get('grid', False)
    rbf = kwargs.get('rbf', False)
    bigaus = kwargs.get('bigaus', False)
    dataset_name = kwargs.get('dataset_name')
    X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test, classLatMedian, classLonMedian, userLocation, loc_train = data
    input_size = X_train.shape[1]
    output_size = Y_train.shape[1] if len(Y_train.shape) == 2 else np.max(Y_train) + 1
    batch_size = 500 if X_train.shape[0] < 10000 else 10000
    mus = None
    if rbf or bigaus:
        logging.info('initializing mus by clustering training points')
        mus = get_cluster_centers(loc_train, n_cluster=hid_size)
        logging.info('first mu is %s' %str(mus[0, :]))
    model = NNModel(n_epochs=1000, batch_size=batch_size, regul_coef=regul, 
                    input_size=input_size, output_size=output_size, hidden_layer_sizes=[hid_size, hid_size, hid_size], 
                    drop_out=True, dropout_coef=dropout_coef, early_stopping_max_down=10, 
                    input_sparse=True, reload=False, rbf=rbf, bigaus=bigaus, mus=mus)

    model.fit(X_train, Y_train, X_dev, Y_dev, X_test, Y_test)
    y_pred = model.predict(X_test)
    geo_eval(Y_test, y_pred, U_test, classLatMedian, classLonMedian, userLocation)
    

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
        '-drop','--dropout', metavar='float',
        help='dropout coef default 0.5',
        type=float, default=0.5)
    parser.add_argument(
        '-cel','--celebrity', metavar='int',
        help='celebrity threshold',
        type=int, default=10)
    
    parser.add_argument(
        '-conv', '--convolution', action='store_true',
        help='if true do convolution')
    parser.add_argument(
        '-map', '--map', action='store_true',
        help='if true just draw maps from pre-trained model')
    parser.add_argument(
        '-tune', '--tune', action='store_true',
        help='if true tune the hyper-parameters')   
    parser.add_argument(
        '-tf', '--tensorflow', action='store_true',
        help='if exists run with tensorflow') 
    parser.add_argument(
        '-autoencoder', '--autoencoder', action='store_true',
        help='if exists adds autoencoder to NN') 
    parser.add_argument(
        '-grid', '--grid', action='store_true',
        help='if exists transforms the input from lat/lon to distance from grids on map') 
    parser.add_argument(
        '-rbf', '--rbf', action='store_true',
        help='if exists transforms the input from lat/lon to rbf probabilities and learns centers and sigmas as well.') 
    parser.add_argument(
        '-bigaus', '--bigaus', action='store_true',
        help='if exists transforms the input from lat/lon to bivariate gaussian probabilities and learns centers and sigmas as well.') 
    parser.add_argument(
        '-m', '--message', type=str) 
    args = parser.parse_args(argv)
    return args
if __name__ == '__main__':
    #nice -n 10 python loc2lang.py -d ~/datasets/na/processed_data/ -enc utf-8 -reg 0 -drop 0.0 -mindf 200 -hid 1000 -bigaus -autoencoder -map
    args = parse_args(sys.argv[1:])
    datadir = args.dir
    dataset_name = datadir.split('/')[-3]
    logging.info('dataset: %s' % dataset_name)
    data = load_data(data_home=args.dir, encoding=args.encoding, mindf=args.mindf, grid=args.grid, dataset_name=dataset_name)
    train(data, regul_coef=args.regularization, dropout_coef=args.dropout, 
          hidden_size=args.hidden, autoencoder=args.autoencoder, grid=args.grid, rbf=args.rbf, bigaus=args.bigaus, dataset_name=dataset_name)
