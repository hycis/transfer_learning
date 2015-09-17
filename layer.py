
from mozi.layers.linear import Linear
from mozi.layers.template import Template
from mozi.layers.activation import RELU


class Merge(Template):

    def __init__(self, input_dim, output_dim):
        self.layers = []
        self.layers.append(RELU())
        self.layers.append(Linear(input_dim,4096))
        self.layers.append(RELU())
        self.layers.append(Linear(4096, output_dim))
        self.params = []
        for layer in self.layers:
            self.params += layers.params

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
        left = state_below[0]
        left, _ = self.left_model.test_fprop(left)
        right = state_below[1]
        right, _ = self.right_model.test_fprop(right)
        return left, right

    def _train_fprop(self, state_below):
        left = state_below[0]
        left, _ = self.left_model.train_fprop(left)
        right = state_below[1]
        right, _ = self.right_model.train_fprop(right)
        return left, right
