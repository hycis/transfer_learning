from layer import *
from mozi.layers.activation import RELU, Softmax, Sigmoid
from mozi.layers.normalization import LRN
from mozi.layers.convolution import Convolution2D, Pooling2D
from mozi.layers.linear import Linear
from mozi.layers.noise import Dropout
from mozi.layers.misc import Flatten
from mozi.cost import entropy, error
from mozi.model import Sequential
from mozi.learning_method import SGD
import theano.tensor as T
from mozi.datasets.dataset import MultiInputsData, SingleBlock
import numpy as np
from mozi.train_object import TrainObject

import os


def setenv():
    NNdir = os.path.dirname(os.path.realpath(__file__))

    # directory to save all the dataset
    os.environ['MOZI_DATA_PATH'] = NNdir + '/data'
    # directory for saving the database that is used for logging the results
    os.environ['MOZI_DATABASE_PATH'] = NNdir + '/database'
    # directory to save all the trained models and outputs
    os.environ['MOZI_SAVE_PATH'] = NNdir + '/save'

    print('MOZI_DATA_PATH = ' + os.environ['MOZI_DATA_PATH'])
    print('MOZI_SAVE_PATH = ' + os.environ['MOZI_SAVE_PATH'])
    print('MOZI_DATABASE_PATH = ' + os.environ['MOZI_DATABASE_PATH'])



def _left_model(text_input_dim, merged_dim):
    left_model = Sequential(input_var=T.matrix(), output_var=T.matrix())
    left_model.add(Linear(text_input_dim, 100))
    left_model.add(RELU())
    left_model.add(Linear(100, merged_dim))
    return left_model


def _right_model(img_input_dim, merged_dim):
    c, h, w = img_input_dim

    valid = lambda x, y, kernel, stride : ((x-kernel)/stride + 1, (y-kernel)/stride + 1)
    full = lambda x, y, kernel, stride : ((x+kernel)/stride - 1, (y+kernel)/stride - 1)

    right_model = Sequential(input_var=T.tensor4(), output_var=T.matrix())
    right_model.add(Convolution2D(input_channels=3, filters=8, kernel_size=(3,3), stride=(1,1), border_mode='full'))
    h, w = full(h, w, 3, 1)
    right_model.add(RELU())
    right_model.add(Convolution2D(input_channels=8, filters=8, kernel_size=(3,3), stride=(1,1), border_mode='valid'))
    h, w = valid(h, w, 3, 1)
    right_model.add(RELU())
    right_model.add(Pooling2D(poolsize=(2, 2), stride=(1,1), mode='max'))
    h, w = valid(h, w, 2, 1)
    right_model.add(Dropout(0.25))

    right_model.add(Convolution2D(input_channels=8, filters=8, kernel_size=(3,3), stride=(1,1), border_mode='full'))
    h, w = full(h, w, 3, 1)
    right_model.add(RELU())
    right_model.add(Convolution2D(input_channels=8, filters=8, kernel_size=(3,3), stride=(1,1), border_mode='valid'))
    h, w = valid(h, w, 3, 1)
    right_model.add(RELU())
    right_model.add(Pooling2D(poolsize=(2, 2), stride=(1,1), mode='max'))
    h, w = valid(h, w, 2, 1)
    right_model.add(Dropout(0.25))

    right_model.add(Flatten())
    right_model.add(Linear(8*h*w, 512))
    right_model.add(Linear(512, 512))
    right_model.add(RELU())
    right_model.add(Dropout(0.5))

    right_model.add(Linear(512, merged_dim))
    return right_model


def train():
    _TEXT_INPUT_DIM_ = 10
    _NUM_EXP_ = 1000
    _IMG_INPUT_DIM_ = (3, 32, 32)
    _OUTPUT_DIM_ = 100
    _TEXT_OUTPUT_DIM_ = 100
    _IMG_OUTPUT_DIM_ = 80

    # build dataset
    txt = np.random.rand(_NUM_EXP_, _TEXT_INPUT_DIM_)
    img = np.random.rand(_NUM_EXP_, *_IMG_INPUT_DIM_)
    y = np.random.rand(_NUM_EXP_, _OUTPUT_DIM_)
    data = MultiInputsData(X=(txt, img), y=y)

    # build left and right model
    left_model = _left_model(_TEXT_INPUT_DIM_, _TEXT_OUTPUT_DIM_)
    right_model = _right_model(_IMG_INPUT_DIM_, _IMG_OUTPUT_DIM_)

    # build the master model
    model = Sequential(input_var=(T.matrix(), T.tensor4()), output_var=T.matrix())
    model.add(Parallel(left_model, right_model))
    model.add(FlattenAll())
    model.add(Concate(_TEXT_OUTPUT_DIM_ + _IMG_OUTPUT_DIM_, _OUTPUT_DIM_))

    # build learning method
    learning_method = SGD(learning_rate=0.01, momentum=0.9,
                          lr_decay_factor=0.9, decay_batch=5000)

    # put everything into the train object
    train_object = TrainObject(model = model,
                               log = None,
                               dataset = data,
                               train_cost = entropy,
                               valid_cost = error,
                               learning_method = learning_method,
                               stop_criteria = {'max_epoch' : 10,
                                                'epoch_look_back' : 5,
                                                'percent_decrease' : 0.01}
                               )
    # finally run the code
    train_object.setup()
    train_object.run()



if __name__ == '__main__':
    setenv()
    train()
