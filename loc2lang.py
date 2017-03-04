'''
Created on 21 Feb 2017
Given a training set of lat/lon as input and probability distribution over words as output,
train a model that can predict words based on location.
then try to visualise borders and regions (e.g. try many lat/lon as input and get the probability of word yinz
in the output and visualise that).
@author: af
'''
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
import operator
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
from sklearn.preprocessing import normalize
from haversine import haversine
from _collections import defaultdict
from scipy import stats
from twokenize import tokenize
from mpl_toolkits.basemap import Basemap, cm
from scipy.interpolate import griddata as gd
from lasagne_layers import SparseInputDenseLayer, GaussianRBFLayer, BivariateGaussianLayer
from shapely.geometry import MultiPoint, Point, Polygon
import shapefile
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
                 bigaus=False):
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
                l_hid = GaussianRBFLayer(l_in, num_units=self.hidden_layer_sizes[0])
            elif self.bigaus:
                logging.info('adding diagonal bivariate gaussian layer...')
                l_hid = BivariateGaussianLayer(l_in, num_units=self.hidden_layer_sizes[0])
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
        if len(self.hidden_layer_sizes) > 1 and not self.rbf:
            for h_size in self.hidden_layer_sizes[1:]:
                l_hid = lasagne.layers.DenseLayer(l_hid, num_units=h_size, 
                                                   nonlinearity=lasagne.nonlinearities.softmax,
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
        
        #self.pred = self.output.argmax(-1)
        loss = lasagne.objectives.squared_error(self.output, self.Y_sym)
        loss = loss.mean()
        eval_loss = lasagne.objectives.squared_error(self.eval_output, self.Y_sym)
        eval_loss = eval_loss.mean() 
        
        

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
            l1_share_out = 0.5
            l1_share_hid = 0.5
            regul_coef_out, regul_coef_hid = self.regul_coef, self.regul_coef
            logging.info('regul coefficient for output and hidden lasagne_layers is ' + str(self.regul_coef))
            l1_penalty = lasagne.regularization.regularize_layer_params(self.l_out, l1) * regul_coef_out * l1_share_out
            l2_penalty = lasagne.regularization.regularize_layer_params(self.l_out, l2) * regul_coef_out * (1-l1_share_out)
            if not self.rbf:
                l1_penalty += lasagne.regularization.regularize_layer_params(l_hid, l1) * regul_coef_hid * l1_share_hid
                l2_penalty += lasagne.regularization.regularize_layer_params(l_hid, l2) * regul_coef_hid * (1-l1_share_hid)
            loss = loss + l1_penalty + l2_penalty
            eval_loss = eval_loss + l1_penalty + l2_penalty

        
        parameters = lasagne.layers.get_all_params(self.l_out, trainable=True)
        updates = lasagne.updates.adamax(loss, parameters, learning_rate=2e-3, beta1=0.9, beta2=0.999, epsilon=1e-8)
        self.f_train = theano.function([self.X_sym, self.Y_sym], loss, updates=updates, on_unused_input='warn')
        self.f_val = theano.function([self.X_sym, self.Y_sym], eval_loss, on_unused_input='warn')
        self.f_predict_proba = theano.function([self.X_sym], self.eval_output, on_unused_input='warn')   
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
        model_file = './data/loc2lang_model_' + str(X_train.shape)  + 'encoder_' + \
        str(self.autoencoder) + '_sparse_' + str(self.sparse) +  '_rbf_' + str(self.rbf) + '.pkl'
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
                l_train = self.f_train(x_batch, y_batch)
            l_val = self.f_val(X_dev, Y_dev)
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
    
def load_data(data_home, **kwargs):
    bucket_size = kwargs.get('bucket', 300)
    encoding = kwargs.get('encoding', 'utf-8')
    celebrity_threshold = kwargs.get('celebrity', 10)  
    mindf = kwargs.get('mindf', 10)
    dtype = kwargs.get('dtype', 'float32')
    one_hot_label = kwargs.get('onehot', False)
    grid_transform = kwargs.get('grid', False)
    normalize_words = kwargs.get('norm', True)
    dl = DataLoader(data_home=data_home, bucket_size=bucket_size, encoding=encoding, 
                    celebrity_threshold=celebrity_threshold, one_hot_labels=one_hot_label, 
                    mindf=mindf, maxdf=0.1, norm='l1', idf=True, btf=False, tokenizer=None)
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
    loc_train, W_train, loc_dev, W_dev, loc_test, W_test, vocab = data
    input_size = loc_train.shape[1]
    output_size = W_train.shape[1]
    batch_size = 500 if W_train.shape[0] < 10000 else 10000
    model = NNModel(n_epochs=1000, batch_size=batch_size, regul_coef=regul, 
                    input_size=input_size, output_size=output_size, hidden_layer_sizes=[hid_size, hid_size], 
                    drop_out=True, dropout_coef=dropout_coef, early_stopping_max_down=20, 
                    autoencoder=autoencoder, input_sparse=sp.sparse.issparse(loc_train), reload=False, rbf=rbf)
    #pdb.set_trace()
    model.fit(loc_train, W_train, loc_dev, W_dev, loc_test, W_test)
    
    k = 50
    with open('./data/cities.json', 'r') as fin:
        cities = json.load(fin)
    
    cities = cities[0:10]
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
        logging.info(name)
        logging.info(str(topwords))
    
    
    # us bounding box (-124.848974, 24.396308) - (-66.885444, 49.384358)
    lllat = 24.396308
    lllon = -124.848974
    urlat =  49.384358
    urlon = -66.885444
    lats = np.arange(lllat, urlat, 0.5)
    lons = np.arange(lllon, urlon, 0.5)
    
    check_in_us = True
    coords = []
    for lat in lats:
        for lon in lons:
            if in_us(lat, lon) or not check_in_us:
                coords.append([lat, lon])
    logging.info('%d coords within continental US' %len(coords))
    coords = np.array(coords).astype('float32')
    #grid representation
    if grid_transform:
        grid_coords = grid_representation(input=coords)
    else:
        grid_coords = coords
    preds = model.predict(grid_coords)
    info_file = 'coords-preds-vocab' + str(W_train.shape[0])+ '_' + str(hid_size) + '.pkl'
    logging.info('dumping the results in %s' %info_file)
    with open(info_file, 'wb') as fout:
        pickle.dump((coords, preds, vocab), fout)

    contour_me(info_file)       
    
    
def contour_me(info_file='coords-preds-vocab5685_200.pkl', **kwargs):
    lllat = 24.396308
    lllon = -124.848974
    urlat =  49.384358
    urlon = -66.885444
    fig = plt.figure(figsize=(10, 8))
    grid_transform = kwargs.get('grid', False)
    ax = fig.add_subplot(111, axisbg='w', frame_on=False)
    grid_interpolation_method = 'nearest'
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
    logging.info('reading info...')
    with open(info_file, 'rb') as fin:
        coords, preds, vocab = pickle.load(fin)
    vocabset = set(vocab)
    with open('./data/cities.json', 'r') as fin:
        cities = json.load(fin)
    
    cities = cities[0:30]
    topk_words = []
    for city in cities:
        name = city['city'].lower()
        topk_words.append(name)
    wi = 0
    for word in topk_words:
        if word in vocabset:
            logging.info('%d mapping %s' %(wi, word))
            wi += 1
            index = vocab.index(word)
            scores = preds[:, index]
            m = Basemap(llcrnrlat=lllat,
            urcrnrlat=urlat,
            llcrnrlon=lllon,
            urcrnrlon=urlon,
            resolution='i', projection='cyl')
            m.drawmapboundary(fill_color = 'white')
            m.drawcoastlines()
            m.drawcountries()
            m.drawstates()
              
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
                method=grid_interpolation_method)
            con = m.contourf(xi, yi, zi, cmap=cm.s3pcpn)
            cbar = m.colorbar(con,location='right',pad="3%", format='%.0e')
            plt.setp(cbar.ax.get_yticklabels(), visible=False)
            #cbar.ax.set_yticklabels(['low', 'high'])
            #cbar.ax.tick_params(labelsize=6) 
            cbar.set_label('prob')
            plt.title('term: ' + word )
            plt.savefig('./maps/' + word + '_' + grid_interpolation_method +  '.pdf')

        
    for region, words in region_words.iteritems():
        for word in words:
            if word in vocabset:
                logging.info('mapping %s' % word)
                index = vocab.index(word)
                scores = preds[:, index]
                m = Basemap(llcrnrlat=lllat,
                urcrnrlat=urlat,
                llcrnrlon=lllon,
                urcrnrlon=urlon,
                resolution='i', projection='cyl')
                m.drawmapboundary(fill_color = 'white')
                m.drawcoastlines()
                m.drawcountries()
                m.drawstates()
                
                
    
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
                    method=grid_interpolation_method)
                con = m.contourf(xi, yi, zi, cmap=cm.s3pcpn)
                cbar = m.colorbar(con,location='right',pad="3%", format='%.0e')
                plt.setp(cbar.ax.get_yticklabels(), visible=False)
                #cbar.ax.set_yticklabels(['low', 'high'])
                #cbar.ax.tick_params(labelsize=6) 
                cbar.set_label('prob')
                plt.title('term: ' + word + ' dialect region: ' + region)
                plt.savefig('./maps/' + region + '_' + word + '_' + grid_interpolation_method +  '.pdf')
        
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
    args = parser.parse_args(argv)
    return args
if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    if args.map:
        contour_me(grid=args.grid)
    else:
        data = load_data(data_home=args.dir, encoding=args.encoding, mindf=args.mindf, grid=args.grid)
        train(data, regul_coef=args.regularization, dropout_coef=args.dropout, 
              hidden_size=args.hidden, autoencoder=args.autoencoder, grid=args.grid, rbf=args.rbf, bigaus=args.bigaus)