
from mozi.layers.linear import Linear
from mozi.layers.template import Template
from mozi.layers.activation import RELU, Softmax
import theano.tensor as T

class Merge(Template):

    def __init__(self, input_dim, output_dim):
        self.layers = []
        self.layers.append(RELU())
        self.layers.append(Linear(input_dim,200))
        self.layers.append(RELU())
        self.layers.append(Linear(200, output_dim))
        self.layers.append(Softmax())
        self.params = []
        for layer in self.layers:
            self.params += layer.params


    def _test_fprop(self, state_below):
        left, right = state_below
        for layer in self.layers:
            left = layer._test_fprop(left)
            right = layer._test_fprop(right)
        return left, right


    def _train_fprop(self, state_below):
        left, right = state_below
        for layer in self.layers:
            left = layer._train_fprop(left)
            right = layer._train_fprop(right)
        return left, right


class Concate(Template):

    def __init__(self, input_dim, output_dim):
        self.layers = []
        self.layers.append(RELU())
        self.layers.append(Linear(input_dim,200))
        self.layers.append(RELU())
        self.layers.append(Linear(200, output_dim))
        self.layers.append(Softmax())
        self.params = []
        for layer in self.layers:
            self.params += layer.params

    def _test_fprop(self, state_below):
        left, right = state_below
        concat = T.concatenate([left, right], axis=1)
        for layer in self.layers:
            concat = layer._test_fprop(concat)
        return concat


    def _train_fprop(self, state_below):
        left, right = state_below
        concat = T.concatenate([left, right], axis=1)
        for layer in self.layers:
            concat = layer._train_fprop(concat)
        return concat


class Parallel(Template):

    def __init__(self, left_model, right_model):
        self.left_model = left_model
        self.right_model = right_model
        self.params = []
        for L_layer in self.left_model.layers:
            self.params += L_layer.params
        for R_layer in self.right_model.layers:
            self.params += R_layer.params


    def _test_fprop(self, state_below):
        left, right = state_below
        left, _ = self.left_model.test_fprop(left)
        right, _ = self.right_model.test_fprop(right)
        return left, right

    def _train_fprop(self, state_below):
        left, right = state_below
        left, _ = self.left_model.train_fprop(left)
        right, _ = self.right_model.train_fprop(right)
        return left, right


class FlattenAll(Template):

    def flatten(self, state):
        if T.gt(state.ndim, 2):
            state = state.reshape((state.shape[0], T.prod(state.shape[1:])))
        return state

    def _test_fprop(self, state_below):
        left, right = state_below
        left = self.flatten(left)
        right = self.flatten(right)
        return left, right

    def _train_fprop(self, state_below):
        return self._test_fprop(state_below)
