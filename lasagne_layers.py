'''
Created on 3 Feb 2017

@author: af
'''
import numpy as np
import scipy as sp
import theano
import theano.tensor as T
import lasagne
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
from lasagne.regularization import regularize_layer_params
import theano.sparse as S
from lasagne.layers import DenseLayer, DropoutLayer, Layer
from collections import OrderedDict

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
    def __init__(self, incoming, H=None, **kwargs):
        super(SparseConvolutionDenseLayer, self).__init__(incoming, **kwargs)
        self.H = H
        #self.H = self.add_param(H, (H.shape[0], H.shape[1]), name='H')

        
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

    def __init__(self, incoming, H=None, **kwargs):
        super(ConvolutionDenseLayer, self).__init__(incoming, **kwargs)
        self.H = H
        #self.H = self.add_param(H, (H.shape[0], H.shape[1]), name='H')
    
    def get_output_for(self, input, **kwargs):
        target_indices = kwargs.get('target_indices') 
        activation = T.dot(input, self.W)
        #do the convolution
        activation = S.dot(self.H, activation)

        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        activation = activation[target_indices, :]
        return self.nonlinearity(activation)

class GaussianRBFLayer(Layer):

    def __init__(self, incoming, num_units, **kwargs):
        super(GaussianRBFLayer, self).__init__(incoming, **kwargs)
        #self.H = H
        #self.H = self.add_param(H, (H.shape[0], H.shape[1]), name='H')
        self.name = 'GaussianRBFLayer'
        self.num_units = num_units
        if kwargs.get('mus', None):
            self.mus_init = kwargs.get('mus')
        else:
            self.mus_init = np.random.randn(self.num_units, 2).astype('float32')
        
        self.sigmas_init = np.abs(np.random.randn(self.num_units).reshape((self.num_units,))).astype('float32')
        
        self.mus = self.add_param(self.mus_init, self.mus_init.shape, name='mus')
        self.sigmas = self.add_param(self.sigmas_init, self.sigmas_init.shape, name='sigmas')

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)        
    
    def get_output_for(self, input, **kwargs):
        C = self.mus[np.newaxis, :, :]
        X = input[:, np.newaxis, :]
        difnorm = T.sum((C-X)**2, axis=-1)
        a = T.exp(-difnorm * (self.sigmas**2))
        return a

class BivariateGaussianLayer(Layer):

    def __init__(self, incoming, num_units, **kwargs):
        super(BivariateGaussianLayer, self).__init__(incoming, **kwargs)
        #self.H = H
        #self.H = self.add_param(H, (H.shape[0], H.shape[1]), name='H')
        self.name = 'BivariateGaussianLayer'
        self.num_units = num_units
        if kwargs.get('mus', None):
            self.mus_init = kwargs.get('mus')
        else:
            self.mus_init = np.random.randn(self.num_units, 2).astype('float32')
        
        self.sigmas_init = np.abs(np.random.randn(self.num_units, 2).reshape((self.num_units,2))).astype('float32')
        
        self.mus = self.add_param(self.mus_init, self.mus_init.shape, name='mus')
        self.sigmas = self.add_param(self.sigmas_init, self.sigmas_init.shape, name='sigmas')

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)        
    
    def get_output_for(self, input, **kwargs):
        #make sure sigma is positive and nonzero softplus(x) (0, +inf)
        sigmas = T.nnet.softplus(self.sigmas)
        sigmainvs = 1.0 / sigmas
        sigmainvprods = sigmainvs[:, 0] * sigmainvs[:, 1]
        sigmas2 = sigmas ** 2
        mus = self.mus[np.newaxis, :, :]
        X = input[:, np.newaxis, :]
        diff = (X-mus) ** 2
        diffsigma = diff / sigmas2
        diffsigmanorm = T.sum(diffsigma, axis=-1)
        expterm = T.exp(-0.5 * diffsigmanorm)
        probs = (0.5 / np.pi) * sigmainvprods * expterm
        return probs
        
        
        
        
        difnorm = T.sum((C-X)**2, axis=-1)
        a = T.exp(-difnorm * (self.sigmas**2))
        return a