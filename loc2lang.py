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
from itertools import product
mpl.use('Agg')
import matplotlib.mlab as mlab
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

def get_us_border_polygon():
    
    sf = shapefile.Reader("./data/states/cb_2015_us_state_20m")
    shapes = sf.shapes()
    #shapes[i].points
    fields = sf.fields
    records = sf.records()
    state_polygons = {}
    for i, record in enumerate(records):
        state = record[5]
        points = shapes[i].points
        poly = Polygon(points)
        state_polygons[state] = poly

    return state_polygons

#us border
state_polygons = get_us_border_polygon()   

def in_us(lat, lon):
    p = Point(lon, lat)
    for state, poly in state_polygons.iteritems():
        if poly.contains(p):
            return state
    return None

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
        if self.sparse:
            self.X_sym = S.csr_matrix(name='inputs', dtype=self.dtype)
        else:
            self.X_sym = T.matrix()
        self.Y_sym = T.matrix()
        if self.autoencoder:
            logging.info('autoencoder is on!')
        else:
            logging.info('autoencoder is off!')
        l_in = lasagne.layers.InputLayer(shape=(None, self.input_size),
                                         input_var=self.X_sym)

        if self.sparse:
            l_hid = SparseInputDenseLayer(l_in, num_units=self.hidden_layer_sizes[0], 
                                           nonlinearity=lasagne.nonlinearities.tanh,
                                           W=lasagne.init.GlorotUniform())
        else:
            if self.rbf:
                logging.info('adding rbf layer...')
                l_hid = GaussianRBFLayer(l_in, num_units=self.hidden_layer_sizes[0], mus=self.mus)
            elif self.bigaus:
                logging.info('adding bivariate gaussian layer...')
                l_hid = BivariateGaussianLayer(l_in, num_units=self.hidden_layer_sizes[0], mus=self.mus)
            else:
                l_hid = lasagne.layers.DenseLayer(l_in, num_units=self.hidden_layer_sizes[0], 
                                                   nonlinearity=lasagne.nonlinearities.tanh,
                                               W=lasagne.init.GlorotUniform())
        if self.drop_out and self.dropout_coef > 0:
            l_hid = lasagne.layers.dropout(l_hid, p=self.dropout_coef)
        if self.autoencoder:
            #l_hid = lasagne.layers.GaussianNoiseLayer(l_hid, sigma=0.05)
            self.l_out_autoencoder = lasagne.layers.DenseLayer(l_hid, num_units=self.input_size, 
                                                       nonlinearity=lasagne.nonlinearities.softmax,
                                                       W=lasagne.init.GlorotUniform())
        if len(self.hidden_layer_sizes) > 1:
            for h_size in self.hidden_layer_sizes[1:]:
                l_hid = lasagne.layers.DenseLayer(l_hid, num_units=h_size, 
                                                   nonlinearity=lasagne.nonlinearities.rectify,
                                                   W=lasagne.init.GlorotUniform())
                if self.drop_out and self.dropout_coef > 0:
                    l_hid = lasagne.layers.dropout(l_hid, p=self.dropout_coef)
        self.l_out = lasagne.layers.DenseLayer(l_hid, num_units=self.output_size,
                                          nonlinearity=lasagne.nonlinearities.softmax,
                                          W=lasagne.init.GlorotUniform())
        
        
        self.eval_output = lasagne.layers.get_output(self.l_out, self.X_sym, deterministic=True)
        #self.eval_pred = self.eval_output.argmax(-1)

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
        
        

        if self.autoencoder:
            self.autoencoder_eval_output = lasagne.layers.get_output(self.l_out_autoencoder, self.X_sym, deterministic=True)
            self.autoencoder_output = lasagne.layers.get_output(self.l_out_autoencoder, self.X_sym)
            autoencoder_loss = lasagne.objectives.squared_error(self.X_sym, self.autoencoder_output)
            autoencoder_loss = autoencoder_loss.mean() / 1000.0
            autoencoder_loss_eval = lasagne.objectives.squared_error(self.X_sym, self.autoencoder_eval_output)
            autoencoder_loss_eval = autoencoder_loss_eval.mean() / 1000.0
            eval_loss += autoencoder_loss_eval
            loss += autoencoder_loss
        
        if self.regul_coef:
            l1_share_out = 0.0
            l1_share_hid = 0.5
            regul_coef_out, regul_coef_hid = self.regul_coef, self.regul_coef
            logging.info('regul coefficient for output and hidden lasagne_layers is ' + str(self.regul_coef))
            l1_penalty = lasagne.regularization.regularize_layer_params(self.l_out, l1) * regul_coef_out * l1_share_out
            l2_penalty = lasagne.regularization.regularize_layer_params(self.l_out, l2) * regul_coef_out * (1-l1_share_out)
            if not self.rbf and not self.bigaus:
                l1_penalty += lasagne.regularization.regularize_layer_params(l_hid, l1) * regul_coef_hid * l1_share_hid
                l2_penalty += lasagne.regularization.regularize_layer_params(l_hid, l2) * regul_coef_hid * (1-l1_share_hid)
            loss = loss + l1_penalty + l2_penalty
            eval_loss = eval_loss + l1_penalty + l2_penalty

        
        parameters = lasagne.layers.get_all_params(self.l_out, trainable=True)
        updates = lasagne.updates.adamax(loss, parameters, learning_rate=2e-3, beta1=0.9, beta2=0.999, epsilon=1e-8)
        self.f_train = theano.function([self.X_sym, self.Y_sym], loss, updates=updates, on_unused_input='warn')
        self.f_val = theano.function([self.X_sym, self.Y_sym], eval_loss, on_unused_input='warn')
        self.f_cross_entropy_loss = theano.function([self.X_sym, self.Y_sym], eval_cross_entropy_loss, on_unused_input='warn')
        self.f_predict_proba = theano.function([self.X_sym], self.eval_output, on_unused_input='warn') 
        #self.f_debug_output =  theano.function([self.X_sym], self.debug_output, on_unused_input='warn')
        if self.autoencoder:
            self.f_predict_autoencoder = theano.function([self.X_sym], self.autoencoder_eval_output, on_unused_input='warn')        

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
        model_file = './data/loc2lang_model_' + str(self.output_size) + '_' + str(self.hidden_layer_sizes)  + '_encoder_' + \
        str(self.autoencoder) + '_sparse_' + str(self.sparse) +  '_rbf_' + str(self.rbf) + '_bigaus_' + str(self.bigaus) + '.pkl'
        if self.reload:
            if path.exists(model_file):
                logging.info('loading the model from %s' %model_file)
                with open(model_file, 'rb') as fin:
                    params = pickle.load(fin)
                self.set_params(params)
                return
                    
        logging.info('training with %d n_epochs and  %d batch_size' %(self.n_epochs, self.batch_size))
        best_params = None
        best_val_loss = sys.maxint
        n_validation_down = 0
        
        for step in range(self.n_epochs):
            for batch in self.iterate_minibatches(X_train, Y_train, self.batch_size, shuffle=True):
                x_batch, y_batch = batch
                if sp.sparse.issparse(y_batch): y_batch = y_batch.todense().astype('float32')
                l_train = self.f_train(x_batch, y_batch)
            l_val = self.f_cross_entropy_loss(X_dev, Y_dev)
            if l_val < best_val_loss and (best_val_loss - l_val) > (0.0001 * l_val):
                best_val_loss = l_val
                best_params = lasagne.layers.get_all_param_values(self.l_out)
                logging.info('first mu (%f,%f) first covar (%f, %f, %f)' %(best_params[0][0, 0], best_params[0][0, 1], best_params[1][0, 0], best_params[1][0, 1], best_params[2][0]))
                logging.info('second mu (%f,%f) second covar (%f, %f, %f)' %(best_params[0][1, 0], best_params[0][1, 1], best_params[1][1, 0], best_params[1][1, 1], best_params[2][1]))
                n_validation_down = 0
            else:
                n_validation_down += 1
                if n_validation_down > self.early_stopping_max_down:
                    logging.info('validation results went down. early stopping ...')
                    break
            logging.info('iter %d, train loss %f, dev loss %f, best dev loss %f, num_down %d' %(step, l_train, l_val, best_val_loss, n_validation_down))
        lasagne.layers.set_all_param_values(self.l_out, best_params)
        l_test = self.f_cross_entropy_loss(X_test, Y_test)
        logging.info('test loss is %f and perplexity is %f' %(l_test, np.power(2, l_test)))
        logging.info('dumping the model...')
        with open(model_file, 'wb') as fout:
            pickle.dump(best_params, fout)
                
    def predict(self, X):
        prob_dist = self.f_predict_proba(X)
        return prob_dist       

def grid_representation(input, grid_size=0.5):
    lllat = 24.396308
    lllon = -124.848974
    urlat =  49.384358
    urlon = -66.885444
    lats = np.arange(lllat, urlat, grid_size)
    lons = np.arange(lllon, urlon, grid_size)
    num_points = len(lats) * len(lons)
    #X = np.zeros((input.shape[0], num_points), dtype=input.dtype)
    X = sp.sparse.lil_matrix((input.shape[0], num_points), dtype=input.dtype)
    coords = []
    for lat in lats:
        for lon in lons:
            coords.append((lat, lon))

    for sample in range(input.shape[0]):
        latlon = (input[sample, 0], input[sample, 1])
        for i, coord in enumerate(coords):
            dist = haversine(latlon, coord, miles=False)
            if dist < 161:
                X[sample, i] = 1.0 / dist

    #normalize each sample to unit vector
    normalize(X, norm='l1', copy=False)  
    return X.tocsr(copy=False)
          
def norm_words(input):
    '''
    normalize each feature to unit norm, then normalize each sample to unit norm
    '''
    #normalize features to unit norm
    normalize(input, norm='l1', axis=0, copy=False)
    #normalize each sample to unit norm
    normalize(input, norm='l1', axis=1, copy=False)
    return input

def get_cluster_centers(input, n_cluster):
    #kmns = KMeans(n_clusters=n_cluster, n_jobs=10)
    kmns = MiniBatchKMeans(n_clusters=n_cluster, batch_size=1000)
    kmns.fit(input)
    return kmns.cluster_centers_.astype('float32')
    
def get_named_entities(documents, mincount=10):
    '''
    given a list of texts find words that more than 
    50% of time start with a capital letter and return them as NE
    '''
    word_count = defaultdict(int)
    word_capital = defaultdict(int)
    NEs = []
    token_pattern = r'(?u)(?<![#@])\b\w+\b'
    tp = re.compile(token_pattern)
    for doc in documents:
        words = tp.findall(doc)
        for word in words:
            if word[0].isupper():
                word_capital[word.lower()] += 1
            word_count[word.lower()] += 1

    for word, count in word_count.iteritems():
        if count < mincount: continue
        capital = word_capital[word]
        percent = float(capital) / count
        if percent > 0.7:
            NEs.append(word)
    return NEs
          
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


    if city_stops:
        logging.info('adding city names to stop words')
        city_names = list(get_us_city_name())
        stop_words.extend(city_names)
    dl = DataLoader(data_home=data_home, bucket_size=bucket_size, encoding=encoding, 
                    celebrity_threshold=celebrity_threshold, one_hot_labels=one_hot_label, 
                    mindf=mindf, maxdf=0.1, norm='l1', idf=True, btf=True, tokenizer=None, subtf=True, stops=stop_words)
    logging.info('loading dataset...')
    dl.load_data()

    
    ne_file = './data/ne_' + dataset_name + '.json'
    if path.exists(ne_file):
        with codecs.open(ne_file, 'r', encoding='utf-8') as fout:
            NEs = json.load(fout)
        NEs = NEs['nes']
    else:
        NEs = get_named_entities(dl.df_train.text.values, mincount=mindf)
        with codecs.open(ne_file, 'w', encoding='utf-8') as fout:
            json.dump({'nes': NEs}, fout)

        
    U_test = dl.df_test.index.tolist()
    U_dev = dl.df_dev.index.tolist()
    U_train = dl.df_train.index.tolist()  
    #locations should be used as input
    loc_test = np.array([[a[0], a[1]] for a in dl.df_test[['lat', 'lon']].values.tolist()], dtype='float32')
    loc_train = np.array([[a[0], a[1]] for a in dl.df_train[['lat', 'lon']].values.tolist()], dtype='float32')
    loc_dev = np.array([[a[0], a[1]] for a in dl.df_dev[['lat', 'lon']].values.tolist()], dtype='float32')
    
    dl.tfidf()
    #words that should be used in the output and be predicted

    W_train = dl.X_train.astype('float32')
    W_dev = dl.X_dev.todense().astype('float32')
    W_test = dl.X_test.todense().astype('float32')

    if normalize_words:
        W_train = norm_words(W_train)
        W_dev = norm_words(W_dev)
        W_test = norm_words(W_test)

    vocab = dl.vectorizer.get_feature_names()

    # every 2d latlon becomes a high dimensional based on distances to the points on a equal-sized grid over US map
    if grid_transform:
        grid_file = 'grid' + str(loc_train.shape[0]) + '.pkl'
        if path.exists(grid_file):
            logging.info('loading grid from %s' %grid_file)
            with open(grid_file, 'rb') as fin:
                loc_train, loc_dev, loc_test = pickle.load(fin)
        else:
            logging.info('transforming lat/lons to grid representation...')
            loc_train = grid_representation(loc_train)
            loc_dev = grid_representation(loc_dev)
            loc_test = grid_representation(loc_test)
            logging.info('transformation  to %d grids finished.' %loc_train.shape[1])
            logging.info('dumping grids to %s' %grid_file)
            with open(grid_file, 'wb') as fout:
                pickle.dump((loc_train, loc_dev, loc_test), fout)
    else:
        logging.info('gridding is off!')

            
            

    
    data = (loc_train, W_train, loc_dev, W_dev, loc_test, W_test, vocab)
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
    loc_train, W_train, loc_dev, W_dev, loc_test, W_test, vocab = data
    input_size = loc_train.shape[1]
    output_size = W_train.shape[1]
    batch_size = 100 if W_train.shape[0] < 10000 else 10000
    mus = None
    if rbf or bigaus:
        logging.info('initializing mus by clustering training points')
        mus = get_cluster_centers(loc_train, n_cluster=hid_size)
        logging.info('first mu is %s' %str(mus[0, :]))
    model = NNModel(n_epochs=1000, batch_size=batch_size, regul_coef=regul, 
                    input_size=input_size, output_size=output_size, hidden_layer_sizes=[hid_size, hid_size], 
                    drop_out=True, dropout_coef=dropout_coef, early_stopping_max_down=10, 
                    autoencoder=autoencoder, input_sparse=sp.sparse.issparse(loc_train), reload=True, rbf=rbf, bigaus=bigaus, mus=mus)
    #pdb.set_trace()
    model.fit(loc_train, W_train, loc_dev, W_dev, loc_test, W_test)
    
    k = 50
    with open('./data/cities.json', 'r') as fin:
        cities = json.load(fin)
    local_word_file = './data/local_words_'  + str(W_train.shape)+ '_' + str(hid_size) + '.txt'
    with codecs.open(local_word_file, 'w', encoding='utf-8') as fout:
        cities = cities[0:100]
        for city in cities:
            name = city['city']
            lat, lon = city['latitude'], city['longitude']
            loc = np.array([[lat, lon]]).astype('float32')
            #grid representation
            if grid_transform:
                loc = grid_representation(input=loc)
            preds = model.predict(loc)
            topword_indices = np.argsort(preds)[0][::-1][:k]
            topwords = [vocab[i] for i in topword_indices]
            #logging.info(name)
            #logging.info(str(topwords))
            fout.write('\n*****%s*****\n' %name)
            fout.write(str(topwords))
            
    
    
    # us bounding box (-124.848974, 24.396308) - (-66.885444, 49.384358)
    lllat = 24.396308
    lllon = -124.848974
    urlat =  49.384358
    urlon = -66.885444
    step = 0.5
    if dataset_name == 'world-final':
        lllat = -90
        lllon = -180
        urlat = 90
        urlon = 180
        step = 0.5
    lats = np.arange(lllat, urlat, step)
    lons = np.arange(lllon, urlon, step)
    
    check_in_us = True if dataset_name != 'world-final' else False
    
    if check_in_us:
        coords = []
        for lat in lats:
            for lon in lons:
                if in_us(lat, lon):
                    coords.append([lat, lon])
                
        logging.info('%d coords within continental US' %len(coords))
        coords = np.array(coords).astype('float32')
    else:
        coords = np.array(map(list, product(lats, lons))).astype('float32')

    #grid representation

    preds = model.predict(coords)
    info_file = 'coords-preds-vocab' + str(W_train.shape[0])+ '_' + str(hid_size) + '.pkl'
    logging.info('dumping the results in %s' %info_file)
    with open(info_file, 'wb') as fout:
        pickle.dump((coords, preds, vocab), fout)

    contour_me(info_file, dataset_name=dataset_name)       
    
def get_local_words(preds, vocab, NEs=[], k=50):
    #normalize the probabilites of each vocab
    normalized_preds = normalize(preds, norm='l1', axis=0)
    entropies = stats.entropy(normalized_preds)
    sorted_indices = np.argsort(entropies)
    sorted_local_words = np.array(vocab)[sorted_indices].tolist()
    filtered_local_words = []
    NEset = set(NEs)
    for word in sorted_local_words:
        if word in NEset: continue
        filtered_local_words.append(word)
    return filtered_local_words[0:k]
   
def contour_me(info_file='coords-preds-vocab1366766_1000.pkl', **kwargs):
    dataset_name = kwargs.get('dataset_name')
    lllat = 24.396308
    lllon = -124.848974
    urlat =  49.384358
    urlon = -66.885444
    if dataset_name == 'world-final':
        lllat = -90
        lllon = -180
        urlat = 90
        urlon = 180
        
    fig = plt.figure(figsize=(10, 8))
    grid_transform = kwargs.get('grid', False)
    ax = fig.add_subplot(111, axisbg='w', frame_on=False)
    grid_interpolation_method = 'cubic'
    logging.info('interpolation: ' + grid_interpolation_method)
    region_words = {
    "the north":['braht','breezeway','bubbler','clout','davenport','euchre','fridge','hotdish','paczki','pop','sack','soda','toboggan','Yooper'],
    "northeast":['brook','cellar','sneaker','soda'],
    "New England":['grinder','packie','rotary','wicked'],
    "Eastern New England":['bulkhead','Cabinet','frappe','hosey','intervale','jimmies','johnnycake','quahog','tonic'],
    "Northern New England":['ayuh','creemee','dooryard','logan','muckle'],
    "The Mid-Atlantic":['breezeway','hoagie','jawn','jimmies','parlor','pavement','shoobie','youze'],
    "New York City Area":['bodega','dungarees','potsy','punchball','scallion','stoop','wedge'],
    "The Midland":['hoosier'],
    "The South":['banquette','billfold','chuck','commode','lagniappe','yankee','yonder'],
    "The West":['davenport','Hella','snowmachine' ]
    }
    map_dir = './maps/' + info_file.split('.')[0] + '/'
    if os.path.exists(map_dir):
        shutil.rmtree(map_dir)
    os.mkdir(map_dir)
    
    
    topk_words = []    
    
    dialect_words = ['jawn', 'paczki', 'euchre', 'brat', 'toboggan', 'brook', 'grinder', 'yall', 'yinz', 'youze', 'hella', 'yeen']
    topk_words.extend(dialect_words)
    custom_words = ['springfield', 'columbia', 'nigga', 'niqqa', 'bamma', 'cooter', 'britches', 'yapper', 'younguns', 'hotdish', 
                    'schnookered', 'bubbler', 'betcha', 'dontcha']
    topk_words.extend(custom_words)
            
    logging.info('reading info...')
    with open(info_file, 'rb') as fin:
        coords, preds, vocab = pickle.load(fin)
    vocabset = set(vocab)
    
    add_local_words = True
    if add_local_words:
        ne_file = './data/ne_' + dataset_name + '.json'
        with codecs.open(ne_file, 'r', encoding='utf-8') as fout:
            NEs = json.load(fout)
        NEs = NEs['nes']
        local_words = get_local_words(preds, vocab, NEs=NEs, k=500)
        logging.info(local_words)
        topk_words.extend(local_words[0:100])
    
    add_cities = False
    if add_cities:
        with open('./data/cities.json', 'r') as fin:
            cities = json.load(fin)
        cities = cities[0:100]
        for city in cities:
            name = city['city'].lower()
            topk_words.append(name)
    wi = 0
    for word in topk_words:
        if word in vocabset:
            logging.info('%d mapping %s' %(wi, word))
            wi += 1
            index = vocab.index(word)
            scores = np.log(preds[:, index])
            
            m = Basemap(llcrnrlat=lllat,
            urcrnrlat=urlat,
            llcrnrlon=lllon,
            urcrnrlon=urlon,
            resolution='i', projection='cyl')
            '''
            m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
            projection='lcc',lat_1=33,lat_2=45,lon_0=-95, resolution='i')
            '''
            m.drawmapboundary(fill_color = 'white')
            #m.drawcoastlines(linewidth=0.2)
            m.drawcountries(linewidth=0.2)
            if dataset_name != 'world-fianl':
                m.drawstates(linewidth=0.2, color='lightgray')
            #m.fillcontinents(color='white', lake_color='#0000ff', zorder=2)
            #m.drawrivers(color='#0000ff')
            #m.drawlsmask(land_color='gray',ocean_color="#b0c4de", lakes=True)
            #m.drawcounties()
            shp_info = m.readshapefile('./data/us_states_st99/st99_d00','states',drawbounds=True, zorder=0)
            printed_names = []
            ax = plt.gca()
            ax.xaxis.set_visible(False) 
            ax.yaxis.set_visible(False) 
            for spine in ax.spines.itervalues(): 
                spine.set_visible(False) 

            state_names_set = set(short_state_names.values())
            mi_index = 0
            wi_index = 0
            for shapedict,state in zip(m.states_info, m.states):
                draw_state_name = True if dataset_name != 'world-fianl' else False
                if shapedict['NAME'] not in state_names_set: continue
                short_name = short_state_names.keys()[short_state_names.values().index(shapedict['NAME'])]
                if short_name in printed_names and short_name not in ['MI', 'WI']: 
                    continue
                if short_name == 'MI':
                    if mi_index != 3:
                        draw_state_name = False
                    mi_index += 1
                if short_name == 'WI':
                    if wi_index != 2:
                        draw_state_name = False
                    wi_index += 1
                    
                # center of polygon
                x, y = np.array(state).mean(axis=0)
                hull = ConvexHull(state)
                hull_points = np.array(state)[hull.vertices]
                x, y = hull_points.mean(axis=0)
                if short_name == 'MD':
                    y = y - 0.5
                    x = x + 0.5
                elif short_name == 'DC':
                    y = y + 0.1
                elif short_name == 'MI':
                    x = x - 1
                elif short_name == 'RI':
                    x = x + 1
                    y = y - 1
                #poly = MplPolygon(state,facecolor='lightgray',edgecolor='black')
                #x, y = np.median(np.array(state), axis=0)
                # You have to align x,y manually to avoid overlapping for little states
                if draw_state_name:
                    plt.text(x+.1, y, short_name, ha="center", fontsize=5)
                #ax.add_patch(poly)
                #pdb.set_trace()
                printed_names += [short_name,] 
            mlon, mlat = m(*(coords[:,1], coords[:,0]))
            # grid data
            numcols, numrows = 1000, 1000
            xi = np.linspace(mlon.min(), mlon.max(), numcols)
            yi = np.linspace(mlat.min(), mlat.max(), numrows)

            xi, yi = np.meshgrid(xi, yi)
            # interpolate
            x, y, z = mlon, mlat, scores
            #pdb.set_trace()
            #zi = griddata(x, y, z, xi, yi)
            zi = gd(
                (mlon, mlat),
                scores,
                (xi, yi),
                method=grid_interpolation_method, rescale=False)

            #Remove the lakes and oceans
            data = maskoceans(xi, yi, zi)
            con = m.contourf(xi, yi, data, cmap=plt.get_cmap('YlOrRd'))
            #con = m.contour(xi, yi, data, 3, cmap=plt.get_cmap('YlOrRd'), linewidths=1)
            #con = m.contour(x, y, z, 3, cmap=plt.get_cmap('YlOrRd'), tri=True, linewidths=1)
            #conf = m.contourf(x, y, z, 3, cmap=plt.get_cmap('coolwarm'), tri=True)
            cbar = m.colorbar(con,location='right',pad="3%")
            #plt.setp(cbar.ax.get_yticklabels(), visible=False)
            #cbar.ax.tick_params(axis=u'both', which=u'both',length=0)
            #cbar.ax.set_yticklabels(['low', 'high'])
            tick_locator = ticker.MaxNLocator(nbins=9)
            cbar.locator = tick_locator
            cbar.update_ticks()
            cbar.ax.tick_params(labelsize=6) 
            cbar.set_label('logprob')
            for line in cbar.lines: 
                line.set_linewidth(20)

            if dataset_name != 'world-final':
                world_shp_info = m.readshapefile('./data/CNTR_2014_10M_SH/Data/CNTR_RG_10M_2014','world',drawbounds=False, zorder=100)
                for shapedict,state in zip(m.world_info, m.world):
                    if shapedict['CNTR_ID'] not in ['CA', 'MX']: continue
                    poly = MplPolygon(state,facecolor='gray',edgecolor='gray')
                    ax.add_patch(poly)
            plt.title('term: ' + word )
            plt.savefig(map_dir + word + '_' + grid_interpolation_method +  '.pdf')
            plt.close()
            del m

  
def visualise_bigaus(params_file, **kwargs):
    with open(params_file, 'rb') as fin:
        params = pickle.load(fin)

    mus, sigmas, sigma12s = params[0], params[1], params[2] 
    dataset_name = kwargs.get('dataset_name')
    lllat = 24.396308
    lllon = -124.848974
    urlat =  49.384358
    urlon = -66.885444
    if dataset_name == 'world-final':
        lllat = -90
        lllon = -180
        urlat = 90
        urlon = 180
    m = Basemap(llcrnrlat=lllat,
    urcrnrlat=urlat,
    llcrnrlon=lllon,
    urcrnrlon=urlon,
    resolution='i', projection='cyl')
    
    m.drawmapboundary(fill_color = 'white')
    #m.drawcoastlines(linewidth=0.2)
    m.drawcountries(linewidth=0.2)
    m.drawstates(linewidth=0.2, color='lightgray')
    #m.fillcontinents(color='white', lake_color='#0000ff', zorder=2)
    #m.drawrivers(color='#0000ff')
    m.drawlsmask(land_color='gray',ocean_color="#b0c4de", lakes=True)
    lllon, lllat = m(lllon, lllat)
    urlon, urlat = m(urlon, urlat)
    numcols, numrows = 1000, 1000
    X = np.linspace(lllon, urlon, numcols)
    Y = np.linspace(lllat, urlat, numrows)
    #X, Y = np.meshgrid(X, Y)
    

    for k in xrange(mus.shape[0]):
        sigmax=sigmas[k][1]
        sigmay=sigmas[k][0]
        mux=mus[k][1]
        muy=mus[k][0]
        sigmaxy = sigma12s[k]
        corxy = 1.0 / (1 + np.abs(sigmaxy))
        sigmaxy = 0
        Z = mlab.bivariate_normal(X, Y, sigmax=sigmax, sigmay=sigmay, mux=mux, muy=muy, sigmaxy=sigmaxy)
        #Z = maskoceans(X, Y, Z)
        pdb.set_trace()
        con = m.contour(X, Y, Z, tri=True)
    plt.savefig('gaus.pdf')
        
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
    parser.add_argument(
        '-vbi', '--vbi', type=str,
        help='if exists load params from vbi file and visualize bivariate gaussians on a map', default=None) 
    args = parser.parse_args(argv)
    return args
if __name__ == '__main__':
    #nice -n 10 python loc2lang.py -d ~/datasets/na/processed_data/ -enc utf-8 -reg 0 -drop 0.0 -mindf 200 -hid 1000 -bigaus -autoencoder -map
    args = parse_args(sys.argv[1:])
    datadir = args.dir
    dataset_name = datadir.split('/')[-3]
    logging.info('dataset: %s' % dataset_name)
    if args.vbi:
        visualise_bigaus(args.vbi, dataset_name=dataset_name)
    elif args.map:
        contour_me(grid=args.grid, dataset_name=dataset_name)
    else:
        data = load_data(data_home=args.dir, encoding=args.encoding, mindf=args.mindf, grid=args.grid, dataset_name=dataset_name)
        train(data, regul_coef=args.regularization, dropout_coef=args.dropout, 
              hidden_size=args.hidden, autoencoder=args.autoencoder, grid=args.grid, rbf=args.rbf, bigaus=args.bigaus, dataset_name=dataset_name)
