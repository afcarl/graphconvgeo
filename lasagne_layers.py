'''
Created on 3 Feb 2017

@author: af
'''
import scipy as sp
import theano
import theano.tensor as T
import lasagne
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
from lasagne.regularization import regularize_layer_params
import theano.sparse as S
from lasagne.layers import DenseLayer, DropoutLayer

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
