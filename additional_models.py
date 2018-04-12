import sys
import os
import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers import Input,merge
from keras import initializers
from keras.utils import vis_utils
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import linear
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten, Dense, Activation, Reshape, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Deconvolution2D, UpSampling2D, MaxPooling2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D
from keras.layers.noise import GaussianNoise
from keras.regularizers import *
from keras.applications.vgg16 import VGG16
#from keras.layers import MinibatchDiscrimination
from models_WGAN import *

def zclass(generator,img_dim, bn_mode,wd,inject_noise,n_classes,noise_dim, model_name="zClass"):

    noise_input = Input(shape=noise_dim, name="noise_input")
    image_input = Input(shape=img_dim, name="image_input")

    generated_image = generator([noise_input,image_input])

    if K.image_dim_ordering() == "th":
        bn_axis = 1
        min_s = min(img_dim[1:])
    else:
        bn_axis = -1
        min_s = min(img_dim[:-1])


    # Get the list of number of conv filters
    # (first layer starts with 64), filters are subsequently doubled
    nb_conv =int(np.floor(np.log(min_s // 4) / np.log(2)))
    list_f = [64 * min(8, (2 ** i)) for i in range(nb_conv)]

    # First conv with 2x2 strides
    x = Convolution2D(list_f[0], 3, 3, subsample=(2, 2), name="disc_conv2d_1",
                      border_mode="same", bias=False, init=conv2D_init,W_regularizer=l2(wd))(generated_image)
    x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
    x = LeakyReLU(0.2)(x)
    # Conv blocks: Conv2D(2x2 strides)->BN->LReLU
    for i, f in enumerate(list_f[1:]):
        name = "disc_conv2d_%s" % (i + 2)
        x = Convolution2D(f, 3, 3, subsample=(2, 2), name=name, border_mode="same", init=conv2D_init,W_regularizer=l2(wd))(x)
        x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
        x = LeakyReLU(0.2)(x)
        #x = Dropout(0.1)(x)

    
    x = GlobalAveragePooling2D()(x)
#    import code
#    code.interact(local=locals())
    x = Dense(noise_dim[0], name='fc1')(x)
    x = LeakyReLU(0.2)(x)
    x = Dense(noise_dim[0], activation='softmax', name='fc2')(x)

    zclass_model = Model(input=[noise_input,image_input],output=[x],name="zClass")
    visualize_model(zclass_model)

    return zclass_model


def z_coerence(generator,img_dim, bn_mode,wd,inject_noise,n_classes,noise_dim, model_name="zClass"):

    image_input = Input(shape=img_dim, name="image_input")

    x = generator.layers[1].output #layers[0] is the input layer
    x = Dense(noise_dim[0], name='fc3')(x)
    x = LeakyReLU(0.2)(x)
    x = Dense(noise_dim[0], name='fc4')(x)

    zclass_model = Model(input=[generator.layers[0].input,image_input],output=[x],name="zClass")
    visualize_model(zclass_model)

    return zclass_model


    #generated_image = K.function([generator.layers[0].input],
#                                  [generator.layers[1].output])
#    generated_image = generator([noise_input,image_input])
