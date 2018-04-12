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
from keras.constraints import unitnorm
from functools import partial
import tensorflow as tf
from normalization import *
import resnet50

def make_trainable(net, value):
    net.trainable = value
    for l in net.layers:
        l.trainable = value

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

def generator_google_mnistM(noise_dim, img_source_dim,img_dest_dim,deterministic,pureGAN,wd,suffix=None):
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
#    shp = np.expand_dims(img_dim[1:],1) # to make shp= (None, 1, 28, 28)  but is not working
    start_dim = int(s / 4)
    if K.image_dim_ordering() == "th":
        input_channels = img_source_dim[0]
        output_channels = img_dest_dim[0]
        reshape_shape = (input_channels, s, s)
        shp=reshape_shape

    else:
        input_channels = img_source_dim[-1]
        output_channels = img_dest_dim[-1]
        reshape_shape = (s, s, input_channels)
        shp=reshape_shape 
    gen_noise_input = Input(shape=noise_dim, name="generator_input")
    gen_image_input = Input(shape=shp, name="generator_image_input")

    # Noise input and reshaping
    x = Dense(5*s*s, input_dim=noise_dim,W_regularizer=l2(wd))(gen_noise_input)
    x = Reshape((5,s,s))(x)
    x = Activation("relu")(x)

    if deterministic: #here I link or not link the noise vector to the whole network
        g = gen_image_input
    elif pureGAN:
        g = x 
    else:
        g = merge([gen_image_input, x], mode='concat',concat_axis=1) # because of concat_axis=1, will it work on tensorflow NHWC too? 

    x1 = Conv2D(64, (3, 3), border_mode='same', kernel_initializer="he_normal",W_regularizer=l2(wd))(g) #convolved by 3x3 filter to get 64x55x35
    x1 = Activation('relu')(x1)

    for i in range(4):
        x = Conv2D(64, (3, 3), border_mode='same', kernel_initializer="he_normal",W_regularizer=l2(wd))(x1)
        x=BatchNormGAN(axis=1)(x)
        x = Activation('relu')(x)
        x = Conv2D(64, (3, 3), border_mode='same',  kernel_initializer="he_normal",W_regularizer=l2(wd))(x)
        x=BatchNormGAN(axis=1)(x)
        x1 = merge([x, x1], mode='sum')
        x1 = Activation('relu')(x1)

    # Last Conv to get the output image
    x1 = Conv2D(output_channels, (1, 1),name="gen_conv2d_final", border_mode='same', kernel_initializer="he_normal",W_regularizer=l2(wd))(x1)
    x1 = Activation('tanh')(x1)
    if suffix is None:
        generator_model = Model(input=[gen_noise_input,gen_image_input], output=[x1], name="generator_google1")
    else:
        generator_model = Model(input=[gen_noise_input,gen_image_input], output=[x1], name="generator_google2")
    visualize_model(generator_model)
    return generator_model


def generator_2048x7x7(noise_dim, img_source_dim,img_dest_dim,deterministic,pureGAN,wd,suffix=None):
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
#    shp = np.expand_dims(img_dim[1:],1) # to make shp= (None, 1, 28, 28)  but is not working
    start_dim = int(s / 4)
    if K.image_dim_ordering() == "th":
        input_channels = img_source_dim[0]
        output_channels = img_dest_dim[0]
        reshape_shape = (input_channels, s, s)
        shp=reshape_shape

    else:
        input_channels = img_source_dim[-1]
        output_channels = img_dest_dim[-1]
        reshape_shape = (s, s, input_channels)
        shp=reshape_shape 
    gen_noise_input = Input(shape=noise_dim, name="generator_input")
    gen_image_input = Input(shape=shp, name="generator_image_input")

    # Noise input and reshaping
    x = Dense(256*s*s, input_dim=noise_dim,W_regularizer=l2(wd))(gen_noise_input)
    x = Reshape((256,s,s))(x)
    x = Activation("relu")(x)

    if deterministic: #here I link or not link the noise vector to the whole network
        g = gen_image_input
    elif pureGAN:
        g = x 
    else:
        g = merge([gen_image_input, x], mode='concat',concat_axis=1) # because of concat_axis=1, will it work on tensorflow NHWC too? 

    x1 = Conv2D(128, (7, 7),strides=(1,1), border_mode='same', kernel_initializer="he_normal",W_regularizer=l2(wd))(g) #convolved by 3x3 filter to get 64x55x35
    x1 = Activation('relu')(x1)
    x1 = BatchNormGAN(axis=1)(x1)
    x1 = Conv2D(2048, (7, 7),strides=(1,1), border_mode='same', kernel_initializer="he_normal",W_regularizer=l2(wd))(x1) #convolved by 3x3 filter to get 64x55x35

    if suffix is None:
        generator_model = Model(input=[gen_noise_input,gen_image_input], output=[x1], name="generator_google1")
    else:
        generator_model = Model(input=[gen_noise_input,gen_image_input], output=[x1], name="generator_google2")
    visualize_model(generator_model)
    return generator_model


def discriminator_google_mnistM(img_dim,wd):

    disc_input = Input(shape=img_dim, name="discriminator_input")
    x = Conv2D(64, (3, 3), strides=(1, 1), name="conv1",border_mode="same", kernel_initializer=RandomNormal(stddev=0.02),kernel_regularizer=l2(wd))(disc_input) 
    x=BatchNormGAN(axis=1)(x)
    x = Dropout(0.1)(x)
    x = LeakyReLU(0.2)(x)
    x = GaussianNoise( sigma=0.2 )(x)
    x = Conv2D(128, (3, 3), strides=(2, 2), name="conv2",border_mode="same", kernel_initializer=RandomNormal(stddev=0.02),kernel_regularizer=l2(wd))(x) 
    x = Dropout(0.2)(x)
#    x = LeakyReLU(0.2)(x)
    x = GaussianNoise( sigma=0.2 )(x)
    x = Conv2D(256, (3, 3), strides=(2, 2), name="conv3",border_mode="same", kernel_initializer=RandomNormal(stddev=0.02),kernel_regularizer=l2(wd))(x) 
    x=BatchNormGAN(axis=1)(x)
    x = Dropout(0.2)(x)
    x = LeakyReLU(0.2)(x)
    x = GaussianNoise( sigma=0.2 )(x)
    x = Conv2D(512, (3, 3), strides=(2, 2), name="conv4",border_mode="same", kernel_initializer=RandomNormal(stddev=0.02),kernel_regularizer=l2(wd))(x) 
    x = Dropout(0.2)(x)
 #   x = LeakyReLU(0.2)(x)
    x = GaussianNoise( sigma=0.2 )(x)
    x = Flatten()(x)
    x = Dense(1, init=RandomNormal(stddev=0.02),activation='sigmoid', name='fc',W_regularizer=l2(wd))(x)
    discriminator_model = Model(input=[disc_input], output=x, name="discriminator_google")
    visualize_model(discriminator_model)
    return discriminator_model

def discriminator_2048x7x7(img_dim,wd,n_classes,disc_type):
    disc_input = Input(shape=img_dim, name="discriminator_input")
    x = Conv2D(128, (7, 7), strides=(1, 1), name="conv1",border_mode="same", kernel_initializer=RandomNormal(stddev=0.02),kernel_regularizer=l2(wd))(disc_input) 
    x=BatchNormGAN(axis=1)(x)
    x = Dropout(0.1)(x)
    x = LeakyReLU(0.2)(x)
    x = GaussianNoise( sigma=0.2 )(x)
    aux = x
    x = Conv2D(1, (3, 3), strides=(1, 1), name="finale_conv",
                      border_mode="same", kernel_initializer=RandomNormal(stddev=0.02),kernel_regularizer=l2(wd))(x)
    aux = Flatten()(aux)
    aux = Dense(n_classes, activation='softmax', name='auxiliary',W_regularizer=l2(wd))(aux)
    x = GlobalAveragePooling2D()(x)
    discriminator_model_domain = Model(input=[disc_input], output=[x], name="discriminator_domain")
    discriminator_model_class = Model(input=[disc_input], output=[aux], name="discriminator_class")

    visualize_model(discriminator_model_domain)
    visualize_model(discriminator_model_class)
    return discriminator_model_domain, discriminator_model_class
	


def discriminator_dcgan(img_dim,wd,n_classes,disc_type):
    min_s = img_dim[1]
    disc_input = Input(shape=img_dim, name="discriminator_input")

    # Get the list of number of conv filters
    # (first layer starts with 64), filters are subsequently doubled
    nb_conv =int(np.floor(np.log(min_s // 4) / np.log(2)))
    list_f = [64 * min(8, (2 ** i)) for i in range(nb_conv)]

    x = Conv2D(list_f[0], (3, 3), strides=(2, 2), name="disc_conv2d_1",
                      border_mode="same", kernel_initializer=RandomNormal(stddev=0.02),kernel_regularizer=l2(wd))(disc_input)
    x=BatchNormalization(axis=1)(x)
    x = LeakyReLU(0.2)(x)
    for i, f in enumerate(list_f[1:]):
        name = "disc_conv2d_%s" % (i + 2)
        x = Conv2D(f, (3, 3), strides=(2, 2), name=name,
                      border_mode="same", kernel_initializer=RandomNormal(stddev=0.02),kernel_regularizer=l2(wd))(x)
        x=BatchNormalization(axis=1)(x)
        x = LeakyReLU(0.2)(x)
      
    x = Conv2D(1, (3, 3), strides=(1, 1), name="finale_conv",
                      border_mode="same", kernel_initializer=RandomNormal(stddev=0.02),kernel_regularizer=l2(wd))(x)
    if disc_type == "nclass_disc":
        aux = Flatten()(x)
        aux = Dense(n_classes, activation='softmax', name='auxiliary',W_regularizer=l2(wd))(aux)
	x = GlobalAveragePooling2D()(x)
        discriminator_model = Model(input=[disc_input], output=[x,aux], name="discriminator")
    elif disc_type == "simple_disc":
        x = GlobalAveragePooling2D()(x)
        discriminator_model = Model(input=[disc_input], output=[x], name="discriminator")
    else:
        print "ERROR, UNKNOWN DISCRIMINATOR"
    visualize_model(discriminator_model)
    return discriminator_model


def discriminator_dcgan_doubled(img_dim,wd,n_classes,disc_type):
    min_s = img_dim[1]
    disc_input = Input(shape=img_dim, name="discriminator_input")

    # Get the list of number of conv filters
    # (first layer starts with 64), filters are subsequently doubled
    nb_conv =int(np.floor(np.log(min_s // 4) / np.log(2)))
    list_f = [64 * min(8, (2 ** i)) for i in range(nb_conv)]

    x = Conv2D(list_f[0], (3, 3), strides=(2, 2), name="disc_conv2d_1",
                      border_mode="same", kernel_initializer=RandomNormal(stddev=0.02),kernel_regularizer=l2(wd))(disc_input)
    x=BatchNormalization(axis=1)(x)
    x = LeakyReLU(0.2)(x)
    for i, f in enumerate(list_f[1:]):
        name = "disc_conv2d_%s" % (i + 2)
        x = Conv2D(f, (3, 3), strides=(2, 2), name=name,
                      border_mode="same", kernel_initializer=RandomNormal(stddev=0.02),kernel_regularizer=l2(wd))(x)
        x=BatchNormalization(axis=1)(x)
        x = LeakyReLU(0.2)(x)
    aux = x
    x = Conv2D(1, (3, 3), strides=(1, 1), name="finale_conv",
                      border_mode="same", kernel_initializer=RandomNormal(stddev=0.02),kernel_regularizer=l2(wd))(x)
    aux = Flatten()(aux)
    aux = Dense(n_classes, activation='softmax', name='auxiliary',W_regularizer=l2(wd))(aux)
    x = GlobalAveragePooling2D()(x)
    discriminator_model_domain = Model(input=[disc_input], output=[x], name="discriminator_domain")
    discriminator_model_class = Model(input=[disc_input], output=[aux], name="discriminator_class")

    visualize_model(discriminator_model_domain)
    visualize_model(discriminator_model_class)
    return discriminator_model_domain, discriminator_model_class

def discriminator_custom(img_dim,wd):
    min_s = img_dim[1]
    disc_input = Input(shape=img_dim, name="discriminator_input")

    # Get the list of number of conv filters
    # (first layer starts with 64), filters are subsequently doubled
    nb_conv =int(np.floor(np.log(min_s // 4) / np.log(2)))
    list_f = [64 * min(8, (2 ** i)) for i in range(nb_conv+1)]

    x = Conv2D(list_f[0], (3, 3), strides=(2, 2), name="disc_conv2d_1",
                      border_mode="same", kernel_initializer=RandomNormal(stddev=0.02),kernel_regularizer=l2(wd))(disc_input)
    x=BatchNormalization(axis=1)(x)
    x = LeakyReLU(0.2)(x)
    for i, f in enumerate(list_f[1:]):
        name = "disc_conv2d_%s" % (i + 2)
        x = Conv2D(f, (3, 3), strides=(2, 2), name=name,
                      border_mode="same", kernel_initializer=RandomNormal(stddev=0.02),kernel_regularizer=l2(wd))(x)
        x=BatchNormalization(axis=1)(x)
        x = LeakyReLU(0.2)(x)
      
    #x = Conv2D(1, (3, 3), strides=(1, 1), name="finale_conv",
    #                  border_mode="same", kernel_initializer=RandomNormal(stddev=0.02),kernel_regularizer=l2(wd))(x)
    #x = GlobalAveragePooling2D()(x)
    x = Dense(1, init=RandomNormal(stddev=0.02), name='fc',W_regularizer=l2(wd))(x)
    discriminator_model = Model(input=[disc_input], output=x, name="discriminator")
    visualize_model(discriminator_model)
    return discriminator_model

def classificator_google_mnistM(img_dim,n_classes,wd):
    input = Input(shape=img_dim, name="classifier_input")
    x = Conv2D(32, (5, 5), strides=(1, 1), name="conv1",border_mode="same", kernel_initializer="he_normal",kernel_regularizer=l2(wd))(input) 
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
    x = Conv2D(48, (5, 5), strides=(1, 1), name="conv2",border_mode="same", kernel_initializer="he_normal",kernel_regularizer=l2(wd))(x) 
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
    x = Flatten()(x)
    x = Dense(100, init="he_normal",activation="relu", name='fc1',W_regularizer=l2(wd))(x)
    x = Dense(100, init="he_normal",activation="relu", name='fc2',W_regularizer=l2(wd))(x)
    x = Dense(n_classes, init="he_normal",activation="softmax", name='fc_softmax',W_regularizer=l2(wd))(x)
    classifier_model = Model(input=input,output=x,name="classifier")
    visualize_model(classifier_model)
    return classifier_model

def classificator_signs_relu(img_dim,n_classes,wd):
    input = Input(shape=img_dim, name="classifier_input")
    x = Conv2D(32, (3, 3), strides=(1, 1), name="conv1_1",border_mode="same", kernel_initializer="he_normal",kernel_regularizer=l2(wd))(input) 
    x =  Activation('relu')(x)
    x = Conv2D(32, (3, 3), strides=(1, 1), name="conv1_2",border_mode="same", kernel_initializer="he_normal",kernel_regularizer=l2(wd))(x) 
    x =  Activation('relu')(x)
    x = Conv2D(32, (3, 3), strides=(1, 1), name="conv1_3",border_mode="same", kernel_initializer="he_normal",kernel_regularizer=l2(wd))(x) 
    x =  Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), name="conv2_1",border_mode="same", kernel_initializer="he_normal",kernel_regularizer=l2(wd))(x) 
    x =  Activation('relu')(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), name="conv2_2",border_mode="same", kernel_initializer="he_normal",kernel_regularizer=l2(wd))(x) 
    x =  Activation('relu')(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), name="conv2_3",border_mode="same", kernel_initializer="he_normal",kernel_regularizer=l2(wd))(x) 
    x =  Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
    x = Conv2D(128, (3, 3), strides=(1, 1), name="conv3_1",border_mode="same", kernel_initializer="he_normal",kernel_regularizer=l2(wd))(x) 
    x =  Activation('relu')(x)
    x = Conv2D(128, (3, 3), strides=(1, 1), name="conv3_2",border_mode="same", kernel_initializer="he_normal",kernel_regularizer=l2(wd))(x) 
    x =  Activation('relu')(x)
    x = Conv2D(128, (3, 3), strides=(1, 1), name="conv3_3",border_mode="same", kernel_initializer="he_normal",kernel_regularizer=l2(wd))(x) 
    x =  Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
    x = Flatten()(x)
    x = Dense(128, init="he_normal",activation="relu", name='fc1',W_regularizer=l2(wd))(x)
    x = Dense(n_classes, init="he_normal",activation="softmax", name='fc_softmax',W_regularizer=l2(wd))(x)
    classifier_model = Model(input=input,output=x,name="classifier")
    visualize_model(classifier_model)
    return classifier_model

def classificator_signs(img_dim,n_classes,wd):
    input = Input(shape=img_dim, name="classifier_input")
    x = Conv2D(32, (3, 3), strides=(1, 1), name="conv1_1",border_mode="same", kernel_initializer="he_normal",kernel_regularizer=l2(wd))(input) 
    x = ELU()(x)
    x = Conv2D(32, (3, 3), strides=(1, 1), name="conv1_2",border_mode="same", kernel_initializer="he_normal",kernel_regularizer=l2(wd))(x) 
    x = ELU()(x)
    x = Conv2D(32, (3, 3), strides=(1, 1), name="conv1_3",border_mode="same", kernel_initializer="he_normal",kernel_regularizer=l2(wd))(x) 
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), name="conv2_1",border_mode="same", kernel_initializer="he_normal",kernel_regularizer=l2(wd))(x) 
    x = ELU()(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), name="conv2_2",border_mode="same", kernel_initializer="he_normal",kernel_regularizer=l2(wd))(x) 
    x = ELU()(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), name="conv2_3",border_mode="same", kernel_initializer="he_normal",kernel_regularizer=l2(wd))(x) 
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
    x = Conv2D(128, (3, 3), strides=(1, 1), name="conv3_1",border_mode="same", kernel_initializer="he_normal",kernel_regularizer=l2(wd))(x) 
    x = ELU()(x)
    x = Conv2D(128, (3, 3), strides=(1, 1), name="conv3_2",border_mode="same", kernel_initializer="he_normal",kernel_regularizer=l2(wd))(x) 
    x = ELU()(x)
    x = Conv2D(128, (3, 3), strides=(1, 1), name="conv3_3",border_mode="same", kernel_initializer="he_normal",kernel_regularizer=l2(wd))(x) 
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
    x = Flatten()(x)
    x = Dense(128, init="he_normal",activation="relu", name='fc1',W_regularizer=l2(wd))(x)
    x = Dense(n_classes, init="he_normal",activation="softmax", name='fc_softmax',W_regularizer=l2(wd))(x)
    classifier_model = Model(input=input,output=x,name="classifier")
    visualize_model(classifier_model)
    return classifier_model

def resnet50classifier(img_dim,n_classes,wd):
    drop=0.5
    model_name="resnet"
    _input = Input(shape=img_dim, name="classificator_input")
    ResNet = resnet50.ResNet50(_input,Shape=img_dim,weights='imagenet')
    make_trainable(ResNet, False)
    x = Dropout(drop)(ResNet.output)
    out = Dense(n_classes, activation='softmax',init="he_normal", name='fc',W_regularizer=l2(wd))(x)
    resnet_model = Model(input=_input, output=out, name=model_name)
    visualize_model(resnet_model)
    return resnet_model

def classificator_2048x7x7(img_dim,n_classes,wd):
    _input = Input(shape=img_dim, name="classificator_input")
    x = Flatten()(_input)
    out = Dense(n_classes, activation='softmax',init="he_normal", name='fc',W_regularizer=l2(wd))(x)
    resnet_model = Model(input=_input, output=out, name="fc_classifier")
    visualize_model(resnet_model)
    return resnet_model

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

def DCGAN_naive2(generator, discriminator, noise_dim, img_source_dim):
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

def reconstructor(generator1, generator2, noise_dim, img_source_dim):
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
    noise_input2 = Input(shape=noise_dim, name="noise_input2")
    image_input = Input(shape=img_source_dim, name="image_input")

    generated_image = generator1([noise_input,image_input])
    reconstructor_output = generator2([noise_input2,generated_image])
    reconstructor = Model(input=[noise_input,image_input,noise_input2],
                  output=reconstructor_output)
    visualize_model(reconstructor)
    return reconstructor


def reconstructorClass(generator1, generator2, classificator, noise_dim, img_source_dim):
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
    noise_input2 = Input(shape=noise_dim, name="noise_input2")
    image_input = Input(shape=img_source_dim, name="image_input")

    generated_image = generator1([noise_input,image_input])
    reconstructor_output = generator2([noise_input2,generated_image])
    recClass_output = classificator(reconstructor_output)
    reconstructor = Model(input=[noise_input,image_input,noise_input2],
                  output=recClass_output)
    visualize_model(reconstructor)
    return reconstructor


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
