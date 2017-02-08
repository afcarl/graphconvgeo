'''
Created on 27 Dec 2016

@author: af
'''
'''
Created on 22 Apr 2016
@author: af
'''

import pdb
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


'''
These sparse classes are copied from https://github.com/Lasagne/Lasagne/pull/596/commits
'''
class SparseInputDenseLayer(DenseLayer):
    def get_output_for(self, input, **kwargs):
        if not isinstance(input, (S.SparseVariable, S.SparseConstant,
                                  S.sharedvar.SparseTensorSharedVariable)):
            raise ValueError("Input for this layer must be sparse")

        activation = S.dot(input, self.W)
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)
class SparseInputDropoutLayer(DropoutLayer):
    def get_output_for(self, input, deterministic=False, **kwargs):
        if not isinstance(input, (S.SparseVariable, S.SparseConstant,
                                  S.sharedvar.SparseTensorSharedVariable)):
            raise ValueError("Input for this layer must be sparse")

        if deterministic or self.p == 0:
            return input
        else:
            # Using Theano constant to prevent upcasting
            one = T.constant(1, name='one')
            retain_prob = one - self.p

            if self.rescale:
                input = S.mul(input, one/retain_prob)

            input_shape = self.input_shape
            if any(s is None for s in input_shape):
                input_shape = input.shape

            return input * self._srng.binomial(input_shape, p=retain_prob,
                                               dtype=input.dtype)
class SparseConvolutionDenseLayer(DenseLayer):
    def __init__(self, incoming, H, **kwargs):
        super(SparseConvolutionDenseLayer, self).__init__(incoming, **kwargs)
        self.H = H
        
    def get_output_for(self, input, **kwargs):
        if not isinstance(input, (S.SparseVariable, S.SparseConstant,
                                  S.sharedvar.SparseTensorSharedVariable)):
            raise ValueError("Input for this layer must be sparse")
        
        activation = S.dot(input, self.W)
        #do the convolution
        activation = S.dot(self.H, activation)

        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)

class ConvolutionDenseLayer(DenseLayer):

    def __init__(self, incoming, H, **kwargs):
        super(ConvolutionDenseLayer, self).__init__(incoming, **kwargs)
        self.H = H
    
    def get_output_for(self, input, **kwargs):
        target_indices = kwargs.get('target_indices') 
        activation = T.dot(input, self.W)
        #do the convolution
        activation = S.dot(self.H, activation)

        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        activation = activation[target_indices, :]
        return self.nonlinearity(activation)



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




class MLPCONV():
    def __init__(self, 
                 n_epochs=10, 
                 batch_size=1000, 
                 init_parameters=None, 
                 complete_prob=False, 
                 add_hidden=True, 
                 regul_coefs=[5e-5, 5e-5], 
                 save_results=False, 
                 hidden_layer_size=None, 
                 drop_out=False, 
                 dropout_coefs=[0.5, 0.5],
                 early_stopping_max_down=100000,
                 loss_name='log',
                 nonlinearity='rectify'):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.init_parameters = init_parameters
        self.complete_prob = complete_prob
        self.add_hidden = add_hidden
        self.regul_coefs = regul_coefs
        self.save_results = save_results
        self.hidden_layer_size = hidden_layer_size
        self.drop_out = drop_out
        self.dropout_coefs = dropout_coefs
        self.early_stopping_max_down = early_stopping_max_down
        self.loss_name = loss_name
        self.nonlinearity = 'rectify'

    def fit(self, X, train_indices, dev_indices, test_indices, Y_train, Y_dev, Y_test, H):
        logging.info('building the network...' + ' hidden:' + str(self.add_hidden))
        in_size = X.shape[1]
        drop_out_hid, drop_out_in = self.dropout_coefs
        if self.complete_prob:
            out_size = Y_train.shape[1]
        else:
            out_size = len(set(Y_train.tolist()))
        logging.info('output size is %d' %out_size)
        
        self.X = X
        self.train_indices = train_indices
        self.dev_indices = dev_indices
        self.test_indices = test_indices
        
        logging.info('input layer size: %d, hidden layer size: %d, output layer size: %d'  %(in_size, self.hidden_layer_size, out_size))
        # Prepare Theano variables for inputs and targets

        self.X_sym = S.csr_matrix(name='inputs', dtype='float32')
        self.target_indices_sym = T.ivector()
        if self.complete_prob:
            self.y_sym = T.matrix()
        else:
            self.y_sym = T.ivector()    

        if self.nonlinearity == 'rectify':
            nonlinearity = lasagne.nonlinearities.rectify
        elif self.nonlinearity == 'sigmoid':
            nonlinearity = lasagne.nonlinearities.sigmoid
        elif self.nonlinearity == 'tanh':
            nonlinearity = lasagne.nonlinearities.tanh
        else:
            nonlinearity = lasagne.nonlinearities.rectify
        
        
        l_in = lasagne.layers.InputLayer(shape=(None, in_size),
                                         input_var=self.X_sym)
        
        if self.drop_out:
            l_in = SparseInputDropoutLayer(l_in, p=drop_out_in)
    
        l_hid1 = SparseConvolutionDenseLayer(l_in, H=H, 
                                            num_units=self.hidden_layer_size,
                                            nonlinearity=nonlinearity,
                                            W=lasagne.init.GlorotUniform()
                                            )
        if self.drop_out:
            l_hid1 = lasagne.layers.dropout(l_hid1, p=drop_out_hid)
        
        self.l_out = ConvolutionDenseLayer(l_hid1, H=H,
                                           num_units=out_size,
                                           nonlinearity=lasagne.nonlinearities.softmax,
                                           )

        
        
    
        self.eval_output = lasagne.layers.get_output(self.l_out, self.X_sym, target_indices=self.target_indices_sym, deterministic=True)
        self.eval_pred = self.eval_output.argmax(-1)
        #self.embedding = lasagne.layers.get_output(l_hid1, self.X_sym, H=H,  deterministic=True)        
        #self.f_get_embeddings = theano.function([self.X_sym], self.embedding)
        self.output = lasagne.layers.get_output(self.l_out, self.X_sym, target_indices=self.target_indices_sym)
        self.pred = self.output.argmax(-1)
        if self.loss_name == 'log':
            loss = lasagne.objectives.categorical_crossentropy(self.output, self.y_sym)
        loss = loss.mean()

        eval_loss = lasagne.objectives.categorical_crossentropy(self.eval_output, self.y_sym)
        eval_loss = eval_loss.mean()
        
        l1_share_out = 0.5
        l1_share_hid = 0.5
        regul_coef_out, regul_coef_hid = self.regul_coefs
        logging.info('regul coefficient for output and hidden layers are ' + str(self.regul_coefs))
        l1_penalty = lasagne.regularization.regularize_layer_params(self.l_out, l1) * regul_coef_out * l1_share_out
        l2_penalty = lasagne.regularization.regularize_layer_params(self.l_out, l2) * regul_coef_out * (1-l1_share_out)

        l1_penalty += lasagne.regularization.regularize_layer_params(l_hid1, l1) * regul_coef_hid * l1_share_hid
        l2_penalty += lasagne.regularization.regularize_layer_params(l_hid1, l2) * regul_coef_hid * (1-l1_share_hid)
        loss = loss + l1_penalty + l2_penalty
        eval_loss = eval_loss + l1_penalty + l2_penalty
        
        if self.complete_prob:
            self.y_sym_one_hot = self.y_sym.argmax(-1)
            self.acc = T.mean(T.eq(self.pred, self.y_sym_one_hot))
            self.eval_ac = T.mean(T.eq(self.eval_pred, self.y_sym_one_hot))
        else:
            self.acc = T.mean(T.eq(self.pred, self.y_sym))
            self.eval_acc = T.mean(T.eq(self.eval_pred, self.y_sym))

        parameters = lasagne.layers.get_all_params(self.l_out, trainable=True)
        
        
        #updates = lasagne.updates.nesterov_momentum(loss, parameters, learning_rate=0.01, momentum=0.9)
        #updates = lasagne.updates.sgd(loss, parameters, learning_rate=0.01)
        #updates = lasagne.updates.adagrad(loss, parameters, learning_rate=0.1, epsilon=1e-6)
        #updates = lasagne.updates.adadelta(loss, parameters, learning_rate=0.1, rho=0.95, epsilon=1e-6)
        updates = lasagne.updates.adamax(loss, parameters, learning_rate=4e-3, beta1=0.9, beta2=0.999, epsilon=1e-8)
        
        self.f_train = theano.function([self.X_sym, self.y_sym, self.target_indices_sym], [loss, self.acc], updates=updates)
        self.f_val = theano.function([self.X_sym, self.y_sym, self.target_indices_sym], [eval_loss, self.eval_acc])
        self.f_predict = theano.function([self.X_sym, self.target_indices_sym], self.eval_pred)
        self.f_predict_proba = theano.function([self.X_sym, self.target_indices_sym], self.eval_output)
        
        

        if self.complete_prob:
            Y_train = Y_train.astype('float32')
            Y_dev = Y_dev.astype('float32')
        else:
            Y_train = Y_train.astype('int32')
            Y_dev = Y_dev.astype('int32')


        #Y = np.vstack((Y_train, Y_dev, Y_test))
        #Y_train = np.zeros(Y.shape)
        #Y_dev = np.zeros(Y.shape)
        #Y_test = np.zeros(Y.shape)
        #Y_train[self.train_indices, :] = Y[self.train_indices, :]
        #Y_dev[self.dev_indices, :] = Y[self.dev_indices, :]
        #Y_test[self.test_indices, :] = Y[self.test_indices, :]
        logging.info('training (n_epochs, batch_size) = (' + str(self.n_epochs) + ', ' + str(self.batch_size) + ')' )
        best_params = None
        best_val_loss = sys.maxint
        best_val_acc = 0.0
        n_validation_down = 0
        for n in xrange(self.n_epochs):
            x_batch, y_batch = self.X, Y_train
            l_train, acc_train = self.f_train(x_batch, y_batch, self.train_indices)
            l_val, acc_val = self.f_val(self.X, Y_dev, self.dev_indices)
            if l_val < best_val_loss:
                best_val_loss = l_val
                best_val_acc = acc_val
                best_params = lasagne.layers.get_all_param_values(self.l_out)
                n_validation_down = 0
            else:
                #early stopping
                n_validation_down += 1
            logging.info('epoch ' + str(n) + ' ,train_loss ' + str(l_train) + ' ,acc ' + str(acc_train) + ' ,val_loss ' + str(l_val) + ' ,acc ' + str(acc_val) + ',best_val_acc ' + str(best_val_acc))
            if n_validation_down > self.early_stopping_max_down:
                logging.info('validation results went down. early stopping ...')
                break
        
        lasagne.layers.set_all_param_values(self.l_out, best_params)
        
        logging.info('***************** final results based on best validation **************')
        l_val, acc_val = self.f_val(self.X, Y_dev, self.dev_indices)
        logging.info('Best dev acc: %f' %(acc_val))
        
    def predict(self, dataset_partition):
        if dataset_partition == 'train':
            indices = self.train_indices
        elif dataset_partition == 'dev':
            indices = self.dev_indices
        elif dataset_partition == 'test':
            indices = self.test_indices
        return self.f_predict(self.X, indices)
    
    def predict_proba(self, dataset_partition):
        if dataset_partition == 'train':
            indices = self.train_indices
        elif dataset_partition == 'dev':
            indices = self.dev_indices
        elif dataset_partition == 'test':
            indices = self.test_indices
        return self.f_predict_proba(self.X, indices)
    
    def accuracy(self, dataset_partition, y_true):
        if dataset_partition == 'train':
            indices = self.train_indices
        elif dataset_partition == 'dev':
            indices = self.dev_indices
        elif dataset_partition == 'test':
            indices = self.test_indices
        _loss, _acc = self.f_val(self.X, y_true, indices)
        return _acc
    
    def score(self, X, dataset_partition, y_true):
        return self.accuracy(X, dataset_partition, y_true)      
    def get_embedding(self, indices):
        pass
        #return self.f_get_embeddings(self.X, indices)
        
        
if __name__ == '__main__':
    pass            
