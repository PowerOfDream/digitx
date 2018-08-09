from keras import layers
from keras import optimizers
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.backend import argmax
from keras.activations import softmax
import char_img_gen as cg 
import numpy as np


def digitx_model(input_shape):
    '''
    a model to classify printed 0-9 and x
    Arguments:
        input_shape -- shape of the input images
    
    Returns:
        model -- a Model() instance in Keras
    '''

    X_input = Input(input_shape, name = 'input')

    X = Conv2D(16, (5, 5), strides = (1, 1), padding = 'same', name = 'conv0')(X_input)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name = 'max_pool0')(X)

    X = Conv2D(32, (3, 3), strides = (1, 1), padding = 'same', name = 'conv1')(X)
    X = BatchNormalization(axis = 3, name = 'bn1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name = 'max_pool1')(X)

    X = Conv2D(64, (3, 3), strides = (1, 1), padding = 'same', name = 'conv2')(X)
    X = BatchNormalization(axis = 3, name = 'bn2')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name = 'max_pool2')(X)

    X = Flatten()(X)
    X = BatchNormalization(axis = 1, name = 'bn3')(X)
    X = Dense(128, activation='relu', name = 'after_flatten')(X)
    X = Dense(64, activation='relu', name = 'after_flatten2')(X)
    X = Dense(11, activation='softmax', name = 'softmax_out', kernel_initializer = glorot_uniform())(X)

    model = Model(inputs = X_input, outputs = X, name = 'digitx_model')

    return model

def digitx_model2(input_shape):
    '''
    same as digitx_model, but use NIN & global average pooling
    '''

    X_input = Input(input_shape, name = 'input')

    X = Conv2D(32, (5, 5), strides = (1, 1), padding = 'same', name = 'conv0')(X_input)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name = 'max_pool0')(X)
    X = Conv2D(16, (1, 1), strides = (1, 1), padding = 'same', name = 'nin0')(X)

    X = Conv2D(32, (3, 3), strides = (1, 1), padding = 'same', name = 'conv1')(X)
    X = BatchNormalization(axis = 3, name = 'bn1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name = 'max_pool1')(X)
    X = Conv2D(16, (1, 1), strides = (1, 1), padding = 'same', name = 'nin1')(X)

    X = Conv2D(32, (3, 3), strides = (1, 1), padding = 'same', name = 'conv2')(X)
    X = BatchNormalization(axis = 3, name = 'bn2')(X)
    X = Activation('relu')(X)

    # arXiv 1312.4400 : generate one feature map for each corresponding category.
    # Network in network, 1x1 Conv shrink the channel number into catagory number
    # output shape (None, 8, 8, 11), each fearure represented by a 8x8 matrix
    X = Conv2D(11, (1, 1), strides = (1, 1), padding = 'same', name = 'nin2')(X)

    # arXiv 1312.4400 : take the average of each feature map, 
    # fed the result vector directly into the softmax layer
    X = GlobalAveragePooling2D()(X)
    X = BatchNormalization(axis = 1, name = 'bn3')(X)
    X = Activation('softmax', name="softmax_out")(X)

    model = Model(inputs = X_input, outputs = X, name = 'digitx_model')

    return model


def create(use_global_average_pooling = False):
    '''
    create the digitx model
    '''
    
    if not use_global_average_pooling:
        model = digitx_model((32,32, 1))
        opt = optimizers.Adam(lr=0.001)
    else:
        model = digitx_model2((32,32, 1))
        opt = optimizers.Adam(lr=0.005)   #global average pooing need more speed!

    model.compile(optimizer = opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model


def train(model, epoch):
    '''
    traing the model
    '''

    for i in range(epoch):
        X_train, Y_train = cg.gen_batch_examples(512, 32, "./font/")
        X_train /= 255.0
        X_train = np.reshape(X_train, (512, 32, 32, 1))
        ohY_train = to_categorical(Y_train, num_classes = 11)

        loss_and_metrics = model.train_on_batch(X_train, ohY_train)
        print("At epoch " + str(i+1) + ": " + str(loss_and_metrics))

    return model

def get_feature_maps(model, data):
    '''
    get feature maps from the output of layers conv3
    '''

    layer_name = 'conv3'
    int_layer_model = Model(inputs = model.input, outputs = model.get_layer(layer_name).output)
    int_output = int_layer_model.predict(data)

    return int_output

def test(model, example_num, save_error_image):
    '''
    test this model
    '''

    X_test, Y_test = cg.gen_batch_examples(example_num, 32, "./font/")
    X_test /= 255.0
    X_test = np.reshape(X_test, (example_num, 32, 32, 1))
    #ohY_test = to_categorical(Y_test, num_classes = 11)

    Y_predict = model.predict(X_test, batch_size = 512)
    Y_predict = argmax(Y_predict, axis = 1)
    
    import tensorflow as tf
    sess = tf.Session()
    result = Y_predict.eval(session = sess)
    sess.close()

    t = Y_test.reshape((example_num,)).astype(int)
    succ = np.sum((t == result))
    print("Accuracy = " + str(succ / len(Y_test)))

    if (save_error_image):
        import os
        if not os.path.exists("./tmp"):
            os.makedirs("./tmp")
        for i in range(len(Y_test)):
            if (Y_test[i] != result[i]):
                fileName = "./tmp/" + str(int(Y_test[i, 0])) + "-" + str(int(result[i])) + "-" + str(i) + ".png"
                pix = X_test[i,:,:,:]
                pix = (pix.reshape(32, 32) * 255.0).astype(np.uint8)

                from PIL import Image
                Image.fromarray(pix, 'L').save(fileName)

    #fea_maps = get_feature_maps(model, X_test)
    #return fea_maps, Y_test

