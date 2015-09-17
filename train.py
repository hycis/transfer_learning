from layer import Parallel, Merge
from mozi.layers.activation import RELU, Softmax
from mozi.layers.normalization import LRN
from mozi.layers.convolution import Convolution2D, Pooling2D
from mozi.layers.linear import Linear
from mozi.layers.noise import Dropout
from mozi.layers.misc import Flatten


def _left_model(_TEXT_INPUT_DIM_, _MERGED_DIM_):
    left_model = Sequential(input_var=T.matrix(), output_var=T.matrix())
    left_model.add(Linear(_TEXT_INPUT_DIM_, 1000))
    left_model.add(RELU())
    left_model.add(Linear(1000, 1000))
    return left_model


def _right_model(_IMG_INPUT_DIM_, _MERGED_DIM_):
    c, h, w = _IMG_INPUT_DIM_

    valid = lambda x, y, kernel, stride : ((x-kernel)/stride + 1, (y-kernel)/stride + 1)
    full = lambda x, y, kernel, stride : ((x+kernel)/stride - 1, (y+kernel)/stride - 1)

    right_model = Sequential(input_var=T.tensor4(), output_var=T.matrix())
    right_model.add(Convolution2D(input_channels=3, filters=32, kernel_size=(3,3), stride=(1,1), border_mode='full'))
    h, w = full(h, w, 3, 1)
    right_model.add(RELU())
    right_model.add(Convolution2D(input_channels=32, filters=32, kernel_size=(3,3), stride=(1,1), border_mode='valid'))
    h, w = valid(h, w, 3, 1)
    right_model.add(RELU())
    right_model.add(Pooling2D(poolsize=(2, 2), mode='max'))
    h, w = valid(h, w, 2, 1)
    right_model.add(Dropout(0.25))

    right_model.add(Convolution2D(input_channels=32, filters=64, kernel_size=(3,3), stride=(1,1), border_mode='full'))
    h, w = full(h, w, 3, 1)
    right_model.add(RELU())
    right_model.add(Convolution2D(input_channels=64, filters=64, kernel_size=(3,3), stride=(1,1), border_mode='valid'))
    h, w = valid(h, w, 3, 1)
    right_model.add(RELU())
    right_model.add(Pooling2D(poolsize=(2, 2), mode='max'))
    h, w = valid(h, w, 2, 1)
    right_model.add(Dropout(0.25))

    right_model.add(Flatten())
    right_model.add(Linear(64*h*w, 512))
    right_model.add(RELU())
    right_model.add(Dropout(0.5))

    right_model.add(Linear(512, 1000))
    return right_model


def train():
    _TEXT_INPUT_DIM_ = 10000
    _IMG_INPUT_DIM_ = (3, 220, 220)
    _MERGED_DIM_ = 1000
    _OUTPUT_DIM_ = 1000

    left_model = _left_model(_TEXT_INPUT_DIM_, _MERGED_DIM_)
    right_model = _right_model(_IMG_INPUT_DIM_, _MERGED_DIM_)

    model = Sequential(input_var=(T.matrix(), T.tensor4()), output_var=T.matrix())
    model.add(Parallel(left_model, right_model))
    model.add(Merge(_MERGED_DIM_, _OUTPUT_DIM_))
    model.add(Softmax())


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
