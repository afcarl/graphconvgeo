'''
Created on 21 Feb 2017
Given a training set of lat/lon as input and probability distribution over words as output,
train a model that can predict words based on location.
then try to visualise borders and regions (e.g. try many lat/lon as input and get the probability of word yinz
in the output and visualise that).
@author: af
'''
import argparse
import sys
import pdb
import random
from data import DataLoader
import numpy as np
import sys
from os import path
import scipy as sp
import theano
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
from collections import OrderedDict
from _collections import defaultdict
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
np.random.seed(77)
random.seed(77)

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
                 dtype='float32'):
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
        
        self.build()
        
    def build(self):
        self.X_sym = T.matrix()
        self.Y_sym = T.matrix()
        l_in = lasagne.layers.InputLayer(shape=(None, self.input_size),
                                         input_var=self.X_sym)
        if self.drop_out:
            l_in = lasagne.layers.dropout(l_in, p=self.dropout_coef)
        l_hid = lasagne.layers.DenseLayer(l_in, num_units=self.hidden_layer_sizes[0], 
                                           nonlinearity=lasagne.nonlinearities.rectify,
                                           W=lasagne.init.GlorotUniform())
        if self.drop_out:
            l_hid = lasagne.layers.dropout(l_hid, p=self.dropout_coef)
        if len(self.hidden_layer_sizes) > 1:
            for h_size in self.hidden_layer_sizes[1:]:
                l_hid = lasagne.layers.DenseLayer(l_hid, num_units=h_size, 
                                                   nonlinearity=lasagne.nonlinearities.rectify,
                                                   W=lasagne.init.GlorotUniform())
                if self.drop_out:
                    l_hid = lasagne.layers.dropout(l_hid, p=self.dropout_coef)
        self.l_out = lasagne.layers.DenseLayer(l_hid, num_units=self.output_size,
                                          nonlinearity=lasagne.nonlinearities.softmax,
                                          W=lasagne.init.GlorotUniform())
        
        self.eval_output = lasagne.layers.get_output(self.l_out, self.X_sym, deterministic=True)
        self.eval_pred = self.eval_output.argmax(-1)
        #self.embedding = lasagne.lasagne_layers.get_output(l_hid1, self.X_sym, H=H,  deterministic=True)        
        #self.f_get_embeddings = theano.function([self.X_sym], self.embedding)
        self.output = lasagne.layers.get_output(self.l_out, self.X_sym)
        self.pred = self.output.argmax(-1)
        loss = lasagne.objectives.categorical_crossentropy(self.output, self.Y_sym)
        loss = loss.mean()

        eval_loss = lasagne.objectives.categorical_crossentropy(self.eval_output, self.Y_sym)
        eval_loss = eval_loss.mean()
        
        l1_share_out = 0.5
        l1_share_hid = 0.5
        regul_coef_out, regul_coef_hid = self.regul_coef, self.regul_coef
        logging.info('regul coefficient for output and hidden lasagne_layers is ' + str(self.regul_coef))
        l1_penalty = lasagne.regularization.regularize_layer_params(self.l_out, l1) * regul_coef_out * l1_share_out
        l2_penalty = lasagne.regularization.regularize_layer_params(self.l_out, l2) * regul_coef_out * (1-l1_share_out)

        l1_penalty += lasagne.regularization.regularize_layer_params(l_hid, l1) * regul_coef_hid * l1_share_hid
        l2_penalty += lasagne.regularization.regularize_layer_params(l_hid, l2) * regul_coef_hid * (1-l1_share_hid)
        loss = loss + l1_penalty + l2_penalty
        eval_loss = eval_loss + l1_penalty + l2_penalty
        #self.y_sym_one_hot = self.y_sym.argmax(-1)
        #self.acc = T.mean(T.eq(self.pred, self.y_sym_one_hot))
        #self.eval_ac = T.mean(T.eq(self.eval_pred, self.y_sym_one_hot)) 
        parameters = lasagne.layers.get_all_params(self.l_out, trainable=True)
        updates = lasagne.updates.adam(loss, parameters, learning_rate=4e-3, beta1=0.9, beta2=0.999, epsilon=1e-8)
        self.f_train = theano.function([self.X_sym, self.Y_sym], loss, updates=updates, on_unused_input='warn')
        self.f_val = theano.function([self.X_sym, self.Y_sym], eval_loss, on_unused_input='warn')
        self.f_predict_proba = theano.function([self.X_sym], self.eval_output, on_unused_input='warn')           

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
        logging.info('training with %d n_epochs and  %d batch_size' %(self.n_epochs, self.batch_size))
        best_params = None
        best_val_loss = sys.maxint
        n_validation_down = 0
        for step in range(self.n_epochs):
            for batch in self.iterate_minibatches(X_train, Y_train, self.batch_size, shuffle=True):
                x_batch, y_batch = batch
                l_train = self.f_train(x_batch, y_batch)
            l_val = self.f_val(X_dev, Y_dev)
            
            logging.info('iter %d, train loss %f, dev loss %f, best dev loss %f' %(step, l_train, l_val, best_val_loss))
            if l_val < best_val_loss:
                best_val_loss = l_val
                best_params = lasagne.layers.get_all_param_values(self.l_out)
                n_validation_down = 0
            else:
                n_validation_down += 1
                if n_validation_down > self.early_stopping_max_down:
                    logging.info('validation results went down. early stopping ...')
                    break
        lasagne.layers.set_all_param_values(self.l_out, best_params)
        logging.info('dumping the model...')
        with open('./data/loc2lang_model.pkl', 'wb') as fout:
            pickle.dump(best_params, fout)
                
    def predict(self, X):
        prob_dist = self.f_predict_proba(X)
        return prob_dist       

           
def load_data(data_home, **kwargs):
    bucket_size = kwargs.get('bucket', 300)
    encoding = kwargs.get('encoding', 'utf-8')
    celebrity_threshold = kwargs.get('celebrity', 10)  
    mindf = kwargs.get('mindf', 10)
    dtype = kwargs.get('dtype', 'float64')
    one_hot_label = kwargs.get('onehot', False)
    dl = DataLoader(data_home=data_home, bucket_size=bucket_size, encoding=encoding, 
                    celebrity_threshold=celebrity_threshold, one_hot_labels=one_hot_label, mindf=mindf, norm='l1', idf=True, btf=False)
    dl.load_data()
    U_test = dl.df_test.index.tolist()
    U_dev = dl.df_dev.index.tolist()
    U_train = dl.df_train.index.tolist()  
    #locations should be used as input
    loc_test = np.array([[a[0], a[1]] for a in dl.df_test[['lat', 'lon']].values.tolist()], dtype='float32')
    loc_train = np.array([[a[0], a[1]] for a in dl.df_train[['lat', 'lon']].values.tolist()], dtype='float32')
    loc_dev = np.array([[a[0], a[1]] for a in dl.df_dev[['lat', 'lon']].values.tolist()], dtype='float32')
    dl.tfidf()
    #words that should be used in the output and be predicted

    W_train = dl.X_train.todense().astype('float32')
    W_dev = dl.X_dev.todense().astype('float32')
    W_test = dl.X_test.todense().astype('float32')
    vocab = dl.vectorizer.get_feature_names()
    data = (loc_train, W_train, loc_dev, W_dev, loc_test, W_test, vocab)
    return data
    
def train(data, **kwargs):
    dropout_coef = kwargs.get('dropout_coef', 0.5)
    regul = kwargs.get('regul_coef', 1e-6)
    loc_train, W_train, loc_dev, W_dev, loc_test, W_test, vocab = data
    input_size = 2
    output_size = W_train.shape[1]
    model = NNModel(n_epochs=1000, batch_size=500, regul_coef=regul, 
                    input_size=input_size, output_size=output_size, hidden_layer_sizes=[500, 500], 
                    drop_out=True, dropout_coef=dropout_coef, early_stopping_max_down=10)
    model.fit(loc_train, W_train, loc_dev, W_dev, loc_test, W_test)
    
    k = 100
    cities = [
    {
        "city": "New York", 
        "growth_from_2000_to_2013": "4.8%", 
        "latitude": 40.7127837, 
        "longitude": -74.0059413, 
        "population": "8405837", 
        "rank": "1", 
        "state": "New York"
    }, 
    {
        "city": "Los Angeles", 
        "growth_from_2000_to_2013": "4.8%", 
        "latitude": 34.0522342, 
        "longitude": -118.2436849, 
        "population": "3884307", 
        "rank": "2", 
        "state": "California"
    }, 
    {
        "city": "Chicago", 
        "growth_from_2000_to_2013": "-6.1%", 
        "latitude": 41.8781136, 
        "longitude": -87.6297982, 
        "population": "2718782", 
        "rank": "3", 
        "state": "Illinois"
    }, 
    {
        "city": "Houston", 
        "growth_from_2000_to_2013": "11.0%", 
        "latitude": 29.7604267, 
        "longitude": -95.3698028, 
        "population": "2195914", 
        "rank": "4", 
        "state": "Texas"
    }, 
    {
        "city": "Philadelphia", 
        "growth_from_2000_to_2013": "2.6%", 
        "latitude": 39.9525839, 
        "longitude": -75.1652215, 
        "population": "1553165", 
        "rank": "5", 
        "state": "Pennsylvania"
    }, 
    {
        "city": "Phoenix", 
        "growth_from_2000_to_2013": "14.0%", 
        "latitude": 33.4483771, 
        "longitude": -112.0740373, 
        "population": "1513367", 
        "rank": "6", 
        "state": "Arizona"
    }, 
    {
        "city": "San Antonio", 
        "growth_from_2000_to_2013": "21.0%", 
        "latitude": 29.4241219, 
        "longitude": -98.49362819999999, 
        "population": "1409019", 
        "rank": "7", 
        "state": "Texas"
    }, 
    {
        "city": "San Diego", 
        "growth_from_2000_to_2013": "10.5%", 
        "latitude": 32.715738, 
        "longitude": -117.1610838, 
        "population": "1355896", 
        "rank": "8", 
        "state": "California"
    }, 
    {
        "city": "Dallas", 
        "growth_from_2000_to_2013": "5.6%", 
        "latitude": 32.7766642, 
        "longitude": -96.79698789999999, 
        "population": "1257676", 
        "rank": "9", 
        "state": "Texas"
    }, 
    {
        "city": "San Jose", 
        "growth_from_2000_to_2013": "10.5%", 
        "latitude": 37.3382082, 
        "longitude": -121.8863286, 
        "population": "998537", 
        "rank": "10", 
        "state": "California"
    }, 
    {
        "city": "Austin", 
        "growth_from_2000_to_2013": "31.7%", 
        "latitude": 30.267153, 
        "longitude": -97.7430608, 
        "population": "885400", 
        "rank": "11", 
        "state": "Texas"
    }, 
    {
        "city": "Indianapolis", 
        "growth_from_2000_to_2013": "7.8%", 
        "latitude": 39.768403, 
        "longitude": -86.158068, 
        "population": "843393", 
        "rank": "12", 
        "state": "Indiana"
    }, 
    {
        "city": "Jacksonville", 
        "growth_from_2000_to_2013": "14.3%", 
        "latitude": 30.3321838, 
        "longitude": -81.65565099999999, 
        "population": "842583", 
        "rank": "13", 
        "state": "Florida"
    }, 
    {
        "city": "San Francisco", 
        "growth_from_2000_to_2013": "7.7%", 
        "latitude": 37.7749295, 
        "longitude": -122.4194155, 
        "population": "837442", 
        "rank": "14", 
        "state": "California"
    }]


    for city in cities:
        name = city['city']
        lat, lon = city['latitude'], city['longitude']
        loc = np.array([[lat, lon]]).astype('float32')
        preds = model.predict(loc)
        topword_indices = np.argsort(preds)[0][::-1][:k]
        topwords = [vocab[i] for i in topword_indices]
        logging.info(name)
        logging.info(str(topwords))
    
    
    
    pdb.set_trace()
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
        '-tune', '--tune', action='store_true',
        help='if true tune the hyper-parameters')   
    parser.add_argument(
        '-tf', '--tensorflow', action='store_true',
        help='if exists run with tensorflow') 
    args = parser.parse_args(argv)
    return args
if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    data = load_data(data_home=args.dir, encoding=args.encoding, mindf=args.mindf)
    train(data, regul_coef=args.regularization, dropout_coef=args.dropout)