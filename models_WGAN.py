import sys
import os
import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers import Input,merge
from keras.layers.merge import _Merge
from keras import initializers
from keras.initializers import RandomNormal
from keras.utils import vis_utils
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.activations import linear
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten, Dense, Activation, Reshape, Lambda, Dropout
from keras.layers.convolutional import Conv2D, Convolution2D, UpSampling2D, MaxPooling2D, Conv2DTranspose
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D
from keras.layers.noise import GaussianNoise
from keras.regularizers import *
from keras.applications.vgg16 import VGG16
#from keras.layers import MinibatchDiscrimination THIS LAYER NEEDS TO BE CHECKED ON KERAS 2.0
from keras.constraints import unitnorm
from functools import partial
import tensorflow as tf

def make_trainable(net, value):
    net.trainable = value
    for l in net.layers:
        l.trainable = value

def rm_dropout(model):
    for k in model.layers:
        if type(k) is keras.layers.Dropout:
            model.layers.remove(k)

def rm_Dense(model):
    for k in model.layers:
        if type(k) is keras.layers.Dense:
            model.layers.remove(k)

def conv2D_init(shape, name=None,dim_ordering=None):
    return initializers.random_normal(shape, scale=0.02, name=name)


def wasserstein(y_true, y_pred):
    return K.mean(y_true * y_pred)

def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    gradients = K.gradients(y_pred, averaged_samples)
    gradients = K.concatenate([K.flatten(tensor) for tensor in gradients])
    gradient_l2_norm = K.sqrt(K.sum(K.square(gradients)))
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    return K.mean(y_pred) - K.mean(y_pred) + gradient_penalty

def visualize_model(model):
    model.summary()
    vis_utils.plot_model(model,
                        to_file='./figures/%s.png' % model.name,
                        show_shapes=True,
                        show_layer_names=True)


def append_minibatch_discrimination_features(activation, x, nb_kernels, kernel_dim):
    activation = K.reshape(activation, (-1, nb_kernels, kernel_dim))
    diffs = K.expand_dims(activation, 3) - K.expand_dims(K.permute_dimensions(activation, [1, 2, 0]), 0)
    abs_diffs = K.sum(K.abs(diffs), axis=2)
    minibatch_features = K.sum(K.exp(-abs_diffs), axis=2)
    return K.concatenate([x, minibatch_features], 1)

def generator_upsampling_mnistM(noise_dim, img_source_dim,img_dest_dim, bn_mode,deterministic,pureGAN,inject_noise,wd, model_name="generator_upsampling", dset="mnistM"):
    """DCGAN generator based on Upsampling and Conv2D

    Args:
        noise_dim: Dimension of the noise input
        img_dim: dimension of the image output
        bn_mode: keras batchnorm mode
        model_name: model name (default: {"generator_upsampling"})
        dset: dataset (default: {"mnist"})

    Returns:
        keras model
    """
    s = img_source_dim[1]
    f = 512
#    shp = np.expand_dims(img_dim[1:],1) # to make shp= (None, 1, 28, 28)  but is not working
    start_dim = int(s / 4)
    nb_upconv = 2
    nb_filters = 64
    if K.image_dim_ordering() == "th":
        bn_axis = 1
        input_channels = img_source_dim[0]
        output_channels = img_dest_dim[0]
        reshape_shape = (input_channels, s, s)
        shp=reshape_shape

    else:
        bn_axis = -1
        input_channels = img_source_dim[-1]
        output_channels = img_dest_dim[-1]
        reshape_shape = (s, s, input_channels)
        shp=reshape_shape 
    gen_noise_input = Input(shape=noise_dim, name="generator_input")
    gen_image_input = Input(shape=shp, name="generator_image_input")
    #import code
    #code.interact(local=locals()
    # Noise input and reshaping
    x = Dense(s*s*input_channels, input_dim=noise_dim,W_regularizer=l2(wd))(gen_noise_input)
    x = Reshape(reshape_shape)(x)
    x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)

    x = Activation("relu")(x)
    if deterministic: #here I link or not link the noise vector to the whole network
        g = gen_image_input
    elif pureGAN:
        g = x 
    else:
        g = merge([gen_image_input, x], mode='concat',concat_axis=1) # because of concat_axis=1, will it work on tensorflow too? 

    if inject_noise:
        g = GaussianNoise( sigma=0.02 )(g)
    g_64feats = Convolution2D(nb_filters, 3, 3, border_mode='same', kernel_initializer="he_normal",W_regularizer=l2(wd))(g) #convolved by 3x3 filter to get 64x55x35
    g_64feats = Activation('relu')(g_64feats)

    if inject_noise:
        g_64feats = GaussianNoise( sigma=0.02 )(g_64feats)

    H0 = Convolution2D(nb_filters, 3, 3, border_mode='same', kernel_initializer="he_normal",W_regularizer=l2(wd))(g_64feats)
    H0 = BatchNormalization(mode=bn_mode,axis=1)(H0)  
    H0 = Activation('relu')(H0)

    if inject_noise:
        H0 = GaussianNoise( sigma=0.02 )(H0)
    H0 = Convolution2D(nb_filters, 3, 3, border_mode='same', kernel_initializer="he_normal",W_regularizer=l2(wd))(H0)
    H0 = BatchNormalization(mode=bn_mode,axis=1)(H0)

    H0 = merge([H0, g_64feats], mode='sum')
    H0 = Activation('relu')(H0)
    if inject_noise:
        H0 = GaussianNoise( sigma=0.02 )(H0)

    H1 = Convolution2D(nb_filters, 3, 3, border_mode='same', kernel_initializer="he_normal",W_regularizer=l2(wd))(H0)
    H1 = BatchNormalization(mode=bn_mode,axis=1)(H1)  
    H1 = Activation('relu')(H1)
    if inject_noise:
        H1 = GaussianNoise( sigma=0.02 )(H1)
    H1 = Convolution2D(nb_filters, 3, 3, border_mode='same', kernel_initializer="he_normal",W_regularizer=l2(wd))(H1)
    H1 = BatchNormalization(mode=bn_mode,axis=1)(H1)

    H1 = merge([H0, H1], mode='sum')
    H1 = Activation('relu')(H1)
    if inject_noise:
        H1 = GaussianNoise( sigma=0.02 )(H1)

    H2 = Convolution2D(nb_filters, 3, 3, border_mode='same', kernel_initializer="he_normal",W_regularizer=l2(wd))(H1)
    H2 = BatchNormalization(mode=bn_mode,axis=1)(H2)  
    H2 = Activation('relu')(H2)
    if inject_noise:
        H2 = GaussianNoise( sigma=0.02 )(H2)
    H2 = Convolution2D(nb_filters, 3, 3, border_mode='same', kernel_initializer="he_normal",W_regularizer=l2(wd))(H2)
    H2 = BatchNormalization(mode=bn_mode,axis=1)(H2)
    H2 = merge([H1, H2], mode='sum')
    H2 = Activation('relu')(H2)

    if inject_noise:
        H2 = GaussianNoise( sigma=0.02 )(H2)
    H3 = Convolution2D(nb_filters, 3, 3, border_mode='same', kernel_initializer="he_normal",W_regularizer=l2(wd))(H2)
    H3 = BatchNormalization(mode=bn_mode,axis=1)(H3)  
    H3 = Activation('relu')(H3)
    if inject_noise:
        H3 = GaussianNoise( sigma=0.02 )(H3)
    H3 = Convolution2D(nb_filters, 3, 3, border_mode='same', kernel_initializer="he_normal",W_regularizer=l2(wd))(H3)
    H3 = BatchNormalization(mode=bn_mode,axis=1)(H3)
    H3 = merge([H2, H3], mode='sum')

    if inject_noise:
        H3 = GaussianNoise( sigma=0.02 )(H3)
    H4 = Convolution2D(nb_filters, 3, 3, border_mode='same', kernel_initializer="he_normal",W_regularizer=l2(wd))(H3)
    H4 = BatchNormalization(mode=bn_mode,axis=1)(H4)  
    H4 = Activation('relu')(H4)
    if inject_noise:
        H4 = GaussianNoise( sigma=0.02 )(H4)
    H4 = Convolution2D(nb_filters, 3, 3, border_mode='same', kernel_initializer="he_normal",W_regularizer=l2(wd))(H4)
    H4 = BatchNormalization(mode=bn_mode,axis=1)(H4)
    H4 = merge([H3, H4], mode='sum')

    if inject_noise:
        H4 = GaussianNoise( sigma=0.02 )(H4)
    H5 = Convolution2D(nb_filters, 3, 3, border_mode='same', kernel_initializer="he_normal",W_regularizer=l2(wd))(H4)
    H5 = BatchNormalization(mode=bn_mode,axis=1)(H5)  
    H5 = Activation('relu')(H5)
    if inject_noise:
        H5 = GaussianNoise( sigma=0.02 )(H5)
    H5 = Convolution2D(nb_filters, 3, 3, border_mode='same', kernel_initializer="he_normal",W_regularizer=l2(wd))(H5)
    H5 = BatchNormalization(mode=bn_mode,axis=1)(H5)
    H5 = merge([H4, H5], mode='sum')

    if inject_noise:
        H5 = GaussianNoise( sigma=0.02 )(H5)
    H6 = Convolution2D(nb_filters, 3, 3, border_mode='same', kernel_initializer="he_normal",W_regularizer=l2(wd))(H5)
    H6 = BatchNormalization(mode=bn_mode,axis=1)(H6)  
    H6 = Activation('relu')(H6)
    if inject_noise:
        H6 = GaussianNoise( sigma=0.02 )(H6)
    H6 = Convolution2D(nb_filters, 3, 3, border_mode='same', kernel_initializer="he_normal",W_regularizer=l2(wd))(H6)
    H6 = BatchNormalization(mode=bn_mode,axis=1)(H6)
    H6 = merge([H5, H6], mode='sum')

    if inject_noise:
        H6 = GaussianNoise( sigma=0.02 )(H6)
    H7 = Convolution2D(nb_filters, 3, 3, border_mode='same', kernel_initializer="he_normal",W_regularizer=l2(wd))(H6)
    H7 = BatchNormalization(mode=bn_mode,axis=1)(H7)  
    H7 = Activation('relu')(H7)
    if inject_noise:
        H7 = GaussianNoise( sigma=0.02 )(H7)
    H7 = Convolution2D(nb_filters, 3, 3, border_mode='same', kernel_initializer="he_normal",W_regularizer=l2(wd))(H7)
    H7 = BatchNormalization(mode=bn_mode,axis=1)(H7)
    H7 = merge([H6, H7], mode='sum')

    H11 = Activation('relu')(H7)

    # Last Conv to get the output image
    if inject_noise:
        H11 = GaussianNoise( sigma=0.02 )(H11)
    H11 = Convolution2D(output_channels, 1, 1,name="gen_conv2d_final", border_mode='same', kernel_initializer="he_normal",W_regularizer=l2(wd))(H11)
    g_V = Activation('tanh')(H11)

    generator_model = Model(input=[gen_noise_input,gen_image_input], output=[g_V], name=model_name)
    visualize_model(generator_model)

    return generator_model


def generator_dcgan(noise_dim, img_source_dim,img_dest_dim, bn_mode,deterministic,pureGAN,inject_noise,wd, model_name="generator_dcgan"):
    """DCGAN generator based on Upsampling and Conv2D

    Args:
        noise_dim: Dimension of the noise input
        img_dim: dimension of the image output
        bn_mode: keras batchnorm mode
        model_name: model name (default: {"generator_upsampling"})
        dset: dataset (default: {"mnist"})

    Returns:
        keras model
    """
    s = img_source_dim[1]
    f = 512
#    shp = np.expand_dims(img_dim[1:],1) # to make shp= (None, 1, 28, 28)  but is not working
    start_dim = int(s / 4)
    nb_upconv = 2
    nb_filters = 64
    if K.image_dim_ordering() == "th":
        bn_axis = 1
        input_channels = img_source_dim[0]
        output_channels = img_dest_dim[0]
        reshape_shape = (input_channels, s, s)
        shp=reshape_shape

    else:
        bn_axis = -1
        input_channels = img_source_dim[-1]
        output_channels = img_dest_dim[-1]
        reshape_shape = (s, s, input_channels)
        shp=reshape_shape 
    gen_noise_input = Input(shape=noise_dim, name="generator_input")
    gen_image_input = Input(shape=shp, name="generator_image_input")

    start_dim = int(s / 16)
    n_fc_filters = 16
    x = Dense(n_fc_filters * 16 * 16, input_dim=noise_dim, weight_norm=True,init="he_normal")(gen_noise_input) #WN = True  in AFFINE
    x = Activation("relu")(x)

#    x = Dense(n_fc_filters * 16 * 16, input_dim=noise_dim)(x)
#    x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
#    x = Activation("relu")(x)

    x = Reshape((n_fc_filters,16,16))(x)
    # Upscaling blocks: Upsampling2D->Conv2D->ReLU->BN->Conv2D->ReLU
    for i in range(nb_upconv):
        x = UpSampling2D(size=(2, 2))(x)
        nb_filters = int(f / (2 ** (i + 1)))
        x = Convolution2D(nb_filters, 3, 3, border_mode="same",weight_norm=True, kernel_initializer="he_normal")(x)
 #       x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
        x = Activation("relu")(x)
        x = Convolution2D(nb_filters, 3, 3, border_mode="same",weight_norm=True, kernel_initializer="he_normal")(x)
        x = Activation("relu")(x)

    # Last Conv to get the output image
    x = Convolution2D(output_channels, 3, 3, name="gen_conv2d_final",
                      border_mode="same", activation='tanh', kernel_initializer="he_normal")(x) #W_constraint=unitnorm()

    generator_model = Model(input=[gen_noise_input,gen_image_input], output=[x], name=model_name)
    visualize_model(generator_model)

    return generator_model
#def mdb_layer():
#    if use_mbd:
#        x = Flatten()(x)

#    def minb_disc(x):
#        diffs = K.expand_dims(x, 3) - K.expand_dims(K.permute_dimensions(x, [1, 2, 0]), 0)
#        abs_diffs = K.sum(K.abs(diffs), 2)
#        x = K.sum(K.exp(-abs_diffs), 2)

#        return x

#    def lambda_output(input_shape):
#        return input_shape[:2]

#    num_kernels = 100
#    dim_per_kernel = 5

#    M = Dense(num_kernels * dim_per_kernel, bias=False, activation=None)
#    MBD = Lambda(minb_disc, output_shape=lambda_output)

#    if use_mbd:
#        x_mbd = M(x)
#        x_mbd = Reshape((num_kernels, dim_per_kernel))(x_mbd)
#        x_mbd = MBD(x_mbd)
#       x = merge([x, x_mbd], mode='concat')
#        x = Dense(1, name="disc_dense_1")(x)

def discriminatorResNet(img_dim, bn_mode,model,wd,monsterClass,inject_noise,n_classes, model_name="discriminator",use_mbd=False):

    drop=0.8
    _input = Input(shape=img_dim, name="discriminator_input")
    ResNet = resnet50.ResNet50(_input,Shape=img_dim)
    x = Dropout(drop)(ResNet.output)
    x = Dense(n_classes*2, activation='softmax', name='fc',W_regularizer=l2(wd))(x)
    resnet_model = Model(input=_input, output=x, name=model_name)
    return resnet_model


def discriminator(img_dim, bn_mode,model,wd,monsterClass,inject_noise,n_classes, model_name="discriminator",use_mbd=False):

    if K.image_dim_ordering() == "th":
        bn_axis = 1
        min_s = min(img_dim[1:])
    else:
        bn_axis = -1
        min_s = min(img_dim[:-1])

    disc_input = Input(shape=img_dim, name="discriminator_input")

    # Get the list of number of conv filters
    # (first layer starts with 64), filters are subsequently doubled
    nb_conv =int(np.floor(np.log(min_s // 4) / np.log(2)))
    list_f = [64 * min(8, (2 ** i)) for i in range(nb_conv)]

    # First conv with 2x2 strides
    x = Convolution2D(list_f[0], 3, 3, subsample=(2, 2), name="disc_conv2d_1",
                      border_mode="same", bias=False, kernel_initializer="he_normal",W_regularizer=l2(wd))(disc_input)
    x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)
    # Conv blocks: Conv2D(2x2 strides)->BN->LReLU
    for i, f in enumerate(list_f[1:]):
        name = "disc_conv2d_%s" % (i + 2)
        if inject_noise:
            x = GaussianNoise( sigma=0.02 )(x)
        x = Convolution2D(f, 3, 3, subsample=(2, 2), name=name, border_mode="same", bias=False, kernel_initializer="he_normal",W_regularizer=l2(wd))(x)
        x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
        x = LeakyReLU(0.2)(x)
        x = Dropout(0.3)(x)

    # Last convolution
    if inject_noise:
        x = GaussianNoise( sigma=0.02 )(x)
    aux_feats = Convolution2D(n_classes*2, 3, 3, name="aux_conv", border_mode="same", bias=False, kernel_initializer="he_normal",W_regularizer=l2(wd))(x)
    aux_feats = GlobalAveragePooling2D()(aux_feats)
    aux_feats = LeakyReLU(0.2)(aux_feats)
    x = Convolution2D(1, 3, 3, name="final_conv", border_mode="same", bias=False, kernel_initializer="he_normal",W_regularizer=l2(wd))(x)

    # Average pooling, it serves as traditional GAN single number true/fake
    x = GlobalAveragePooling2D()(x)
    if monsterClass: #2*nClasses (nClasses True, nClasses False) and no true/fake output
        aux = Dense(n_classes*2, activation='softmax', name='auxiliary',W_regularizer=l2(wd))(aux_feats)
        discriminator_model = Model(input=[disc_input], output=aux, name=model_name)
    else:
        aux = Dense(n_classes, activation='softmax', name='auxiliary',W_regularizer=l2(wd))(x)
        discriminator_model = Model(input=[disc_input], output=[x,aux], name=model_name)

    visualize_model(discriminator_model)
    return discriminator_model

def disc1(img_dim, bn_mode,model,wd,monsterClass,inject_noise,n_classes, model_name="discriminator",use_mbd=False):

    if K.image_dim_ordering() == "th":
        bn_axis = 1
        min_s = min(img_dim[1:])
    else:
        bn_axis = -1
        min_s = min(img_dim[:-1])

    disc_input = Input(shape=img_dim, name="discriminator_input")

    # Get the list of number of conv filters
    # (first layer starts with 64), filters are subsequently doubled
    nb_conv =int(np.floor(np.log(min_s // 4) / np.log(2)))
    list_f = [64 * min(8, (2 ** i)) for i in range(nb_conv)]

    # First conv with 2x2 strides
    x = Convolution2D(list_f[0], 3, 3, name="disc_conv2d_1",
                      border_mode="same", bias=False, kernel_initializer="he_normal",W_regularizer=l2(wd))(disc_input)
    x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
    x = LeakyReLU(0.2)(x)
    # Conv blocks: Conv2D(2x2 strides)->BN->LReLU
    for i, f in enumerate(list_f[1:]):
        name = "disc_conv2d_%s" % (i + 2)
        x1 = Convolution2D(64, 3, 3, border_mode='same', kernel_initializer="he_normal",W_regularizer=l2(wd))(x)
        x1 = BatchNormalization(mode=bn_mode,axis=bn_axis)(x1)  
        x1 = LeakyReLU(0.2)(x1)
        x1 = Convolution2D(64, 3, 3, border_mode='same', kernel_initializer="he_normal",W_regularizer=l2(wd))(x1)
        x1 = BatchNormalization(mode=bn_mode,axis=1)(x1)
        x = merge([x1, x], mode='sum')
        x = LeakyReLU(0.2)(x)


    # Last convolution
    aux_feats = Convolution2D(n_classes*2, 3, 3, name="aux_conv", border_mode="same", bias=False, kernel_initializer="he_normal",W_regularizer=l2(wd))(x)
    aux_feats = GlobalAveragePooling2D()(aux_feats)
    aux_feats = LeakyReLU(0.2)(aux_feats)
    x = Convolution2D(1, 3, 3, name="final_conv", border_mode="same", bias=False, kernel_initializer="he_normal",W_regularizer=l2(wd))(x)

    # Average pooling, it serves as traditional GAN single number true/fake
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    if monsterClass: #2*nClasses (nClasses True, nClasses False) and no true/fake output
        aux = Dense(n_classes*2, activation='softmax', name='auxiliary',W_regularizer=l2(wd))(aux_feats)
        discriminator_model = Model(input=[disc_input], output=aux, name=model_name)
    else:
        aux = Dense(n_classes, activation='softmax', name='auxiliary',W_regularizer=l2(wd))(x)
        discriminator_model = Model(input=[disc_input], output=[x,aux], name=model_name)

    visualize_model(discriminator_model)
    return discriminator_model

def discriminator_naive(img_dim, wd,inject_noise, model_name="discriminator"):

    if K.image_dim_ordering() == "th":
        bn_axis = 1
        min_s = min(img_dim[1:])
    else:
        bn_axis = -1
        min_s = min(img_dim[:-1])

    disc_input = Input(shape=img_dim, name="discriminator_input")

    # Get the list of number of conv filters
    # (first layer starts with 64), filters are subsequently doubled
    nb_conv =int(np.floor(np.log(min_s // 4) / np.log(2)))
    list_f = [64 * min(8, (2 ** i)) for i in range(nb_conv)]

    x = Conv2D(list_f[0], (3, 3), strides=(2, 2), name="disc_conv2d_1",
                      border_mode="same",weight_norm=True, kernel_initializer=RandomNormal(stddev=0.02),kernel_regularizer=l2(wd))(disc_input)   
    x = LeakyReLU(0.2)(x)
    for i, f in enumerate(list_f[1:]):
        name = "disc_conv2d_%s" % (i + 2)
        x = Conv2D(f, (3, 3), strides=(2, 2), name=name,
                      border_mode="same",weight_norm=True, kernel_initializer=RandomNormal(stddev=0.02),kernel_regularizer=l2(wd))(x) 
        x = LeakyReLU(0.2)(x)

    x = Conv2D(1, (3, 3), strides=(1, 1), name="finale_conv",
                      border_mode="same",weight_norm=True, kernel_initializer=RandomNormal(stddev=0.02),kernel_regularizer=l2(wd))(x) 
    x = Flatten()(x)
    x = Dense(1, init=RandomNormal(stddev=0.02), name='fc',W_regularizer=l2(wd))(x)
    discriminator_model = Model(input=[disc_input], output=x, name=model_name)
    visualize_model(discriminator_model)
    return discriminator_model



def discriminator_naive1(img_dim, wd,inject_noise, model_name="discriminator"):

    if K.image_dim_ordering() == "th":
        bn_axis = 1
        min_s = min(img_dim[1:])
    else:
        bn_axis = -1
        min_s = min(img_dim[:-1])

    disc_input = Input(shape=img_dim, name="discriminator_input")

    # Get the list of number of conv filters
    # (first layer starts with 64), filters are subsequently doubled
    nb_conv =int(np.floor(np.log(min_s // 4) / np.log(2)))
    list_f = [64 * min(8, (2 ** i)) for i in range(nb_conv)]

    x = Conv2D(list_f[0], (3, 3), strides=(1, 1), name="disc_conv2d_1",
                      border_mode="same",weight_norm=False, kernel_initializer=RandomNormal(stddev=0.02),kernel_regularizer=l2(wd))(disc_input)   
    x = LeakyReLU(0.2)(x)
    for i, f in enumerate(list_f[1:]):
        name = "disc_conv2d_%s" % (i + 2)
        x = Conv2D(f, (3, 3), strides=(2, 2), name=name,
                      border_mode="same",weight_norm=False, kernel_initializer=RandomNormal(stddev=0.02),kernel_regularizer=l2(wd))(x) 
        x = LeakyReLU(0.2)(x)

    x = Conv2D(1, (3, 3), strides=(2, 2), name="finale_conv",
                      border_mode="same",weight_norm=False, kernel_initializer=RandomNormal(stddev=0.02),kernel_regularizer=l2(wd))(x) 
    x = LeakyReLU(0.2)(x)
    x = Flatten()(x)
    x = Dense(1, init=RandomNormal(stddev=0.02), name='fc',W_regularizer=l2(wd))(x)
    discriminator_model = Model(input=[disc_input], output=x, name=model_name)
    visualize_model(discriminator_model)
    return discriminator_model


#Do I need to insert bn_mode? I have to test both bn mode 0 and 2 for resnet too
def resnet(img_dim,n_classes,pretrained,wd, model_name="resnet"):
    drop=0.8
    _input = Input(shape=img_dim, name="discriminator_input")
    if pretrained:
        ResNet = resnet50.ResNet50(_input,Shape=img_dim)
    else:
        ResNet = resnet50.ResNet50(_input,Shape=img_dim,weights='imagenet')
    make_trainable(ResNet, False)
    x = Dropout(drop)(ResNet.output)
    print(drop)
    out = Dense(n_classes, activation='softmax',init="he_normal", name='fc',W_regularizer=l2(wd))(x)
    resnet_model = Model(input=_input, output=out, name=model_name)

    if pretrained:
        model_path = "./models/DCGAN"
        path = os.path.join(model_path, 'VandToVand_5epochs.h5')
        resnet_model.load_weights(path)
    #visualize_model(resnet_model)
    make_trainable(resnet_model, False)
    return resnet_model

#Do I need to insert bn_mode? I have to test both bn mode 0 and 2 for resnet too
def resnet50classifier(img_dim,n_classes,wd):
    drop=0.5
    model_name="resnet"
    _input = Input(shape=img_dim, name="discriminator_input")
    ResNet = resnet50.ResNet50(_input,Shape=img_dim,weights='imagenet')
    make_trainable(ResNet, False)
    x = Dropout(drop)(ResNet.output)
    out = Dense(n_classes, activation='softmax',init="he_normal", name='fc',W_regularizer=l2(wd))(x)
    resnet_model = Model(input=_input, output=out, name=model_name)
    visualize_model(resnet_model)
    return resnet_model

def vgg16(img_dim,n_classes,pretrained,wd, model_name="resnet"):
    drop=0.5
    _input = Input(shape=img_dim, name="discriminator_input")
    vgg16 = VGG16(include_top=False, weights='imagenet')
    x = vgg16(_input)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    out = Dense(n_classes, activation='softmax',init="he_normal", name='fc',W_regularizer=l2(wd))(x)
    vgg16_model = Model(input=_input, output=out, name=model_name)

    model_path = "./models/DCGAN"
    path = os.path.join(model_path, 'vgg16r_OfficeDslrToAmazon.h5')
    vgg16_model.load_weights(path)

    make_trainable(vgg16_model, False)
    return vgg16_model

def GenToClassifierModel(generator, classifier, noise_dim, img_source_dim):
    """GEN + classifier model

    Args:
        generator: keras generator model
        classifier: keras classifier model
        noise_dim: generator input noise dimension
        img_dim: real image data dimension

    Returns:
        keras model
    """
    noise_input = Input(shape=noise_dim, name="noise_input")
    image_input = Input(shape=img_source_dim, name="image_input")

    generated_image = generator([noise_input,image_input])
    y_pred = classifier(generated_image)
    GenToClassifierModel = Model(input=[noise_input,image_input],
                  output=y_pred,
                  name="GenToClassifierModel")
    visualize_model(GenToClassifierModel)
    return GenToClassifierModel

def DCGAN(generator, discriminator, noise_dim, img_source_dim, img_dest_dim,monsterClass):
    """DCGAN generator + discriminator model

    Args:
        generator: keras generator model
        discriminator: keras discriminator model
        noise_dim: generator input noise dimension
        img_dim: real image data dimension

    Returns:
        keras model
    """
    noise_input = Input(shape=noise_dim, name="noise_input")
    image_input = Input(shape=img_source_dim, name="image_input")

    generated_image = generator([noise_input,image_input])
    if monsterClass:
        y_aux = discriminator(generated_image)
        DCGAN = Model(input=[noise_input,image_input],
                  output=y_aux,
                  name="DCGAN")
    else:
        DCGAN_output,y_aux = discriminator(generated_image)
        DCGAN = Model(input=[noise_input,image_input],
                  output=[DCGAN_output,y_aux],
                  name="DCGAN")
    visualize_model(DCGAN)

    return DCGAN


def DCGAN_naive(generator, discriminator, noise_dim, img_source_dim):
    """DCGAN generator + discriminator model

    Args:
        generator: keras generator model
        discriminator: keras discriminator model
        noise_dim: generator input noise dimension
        img_dim: real image data dimension

    Returns:
        keras model
    """
    noise_input = Input(shape=noise_dim, name="noise_input")
    image_input = Input(shape=img_source_dim, name="image_input")

    generated_image = generator([noise_input,image_input])
    DCGAN_output = discriminator(generated_image)
    DCGAN = Model(input=[noise_input,image_input],
                  output=DCGAN_output)
    visualize_model(DCGAN)

    return DCGAN


def generator_deconv(noise_dim, img_source_dim,img_dest_dim, bn_mode,deterministic,pureGAN,inject_noise,wd, model_name="generator_deconv"):
    """DCGAN generator based on Upsampling and Conv2D

    Args:
        noise_dim: Dimension of the noise input
        img_dim: dimension of the image output
        bn_mode: keras batchnorm mode
        model_name: model name (default: {"generator_upsampling"})
        dset: dataset (default: {"mnist"})

    Returns:
        keras model
    """
    s = img_source_dim[1]
    f = 512
#    shp = np.expand_dims(img_dim[1:],1) # to make shp= (None, 1, 28, 28)  but is not working
    start_dim = int(s / 4)
    nb_upconv = 2
    nb_filters = 64
    if K.image_dim_ordering() == "th":
        bn_axis = 1
        input_channels = img_source_dim[0]
        output_channels = img_dest_dim[0]
        reshape_shape = (input_channels, s, s)
        shp=reshape_shape

    else:
        bn_axis = -1
        input_channels = img_source_dim[-1]
        output_channels = img_dest_dim[-1]
        reshape_shape = (s, s, input_channels)
        shp=reshape_shape 
    gen_noise_input = Input(shape=noise_dim, name="generator_input")
    gen_image_input = Input(shape=shp, name="generator_image_input")

    o_shape = (32,256,4,4)
    x = Dense(512 * 4 * 4, input_dim=noise_dim, weight_norm=False)(gen_noise_input) #WN = True  in AFFINE
    x = Activation("relu")(x)
    x = Reshape((512,4,4))(x)
    x = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding="same",weight_norm=False, kernel_initializer="he_normal")(x)
    x = Activation("relu")(x)
    x = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same",weight_norm=False, kernel_initializer="he_normal")(x)
    x = Activation("relu")(x)
    x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same",weight_norm=False, kernel_initializer="he_normal")(x)
    x = Activation("relu")(x)
    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same",weight_norm=False, kernel_initializer="he_normal")(x)
    x = Activation("relu")(x)

    # Last Conv to get the output image
    x = Convolution2D(output_channels, 3, 3, name="gen_conv2d_final",
                      border_mode="same", activation='tanh', kernel_initializer="he_normal")(x) #W_constraint=unitnorm()

    generator_model = Model(input=[gen_noise_input,gen_image_input], output=[x], name=model_name)
    visualize_model(generator_model)

    return generator_model

class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this outputs a random point on the line
    between each pair of input points.
    Inheriting from _Merge is a little messy but it was the quickest solution I could think of.
    Improvements appreciated."""
    def _merge_function(self, inputs):
        weights = K.random_uniform((32, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])

def disc_penalty(discriminator_model, noise_dim, img_source_dim, opt, model_name="disc_penalty_model"):
    image_input_real = Input(shape=img_source_dim, name="image_input_real")
    image_input_gen = Input(shape=img_source_dim, name="image_input_gen")
    averaged_samples = RandomWeightedAverage()([image_input_real, image_input_gen])
    averaged_samples_output = discriminator_model(averaged_samples)
    disc_penalty_model = Model(input=[image_input_real,image_input_gen],
                  output=averaged_samples_output)

    partial_gp_loss = partial(gradient_penalty_loss,
                          averaged_samples=averaged_samples,
                          gradient_penalty_weight=10)
    partial_gp_loss.__name__ = 'gradient_penalty'  # Functions need names or Keras will throw an error

    disc_penalty_model.compile(loss=partial_gp_loss, optimizer=opt)
    return disc_penalty_model
