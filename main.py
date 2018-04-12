# -*- coding: utf-8 -*-
import os
import sys
import time
import numpy as np
import models
from keras.utils import generic_utils
import general_utils
from data_utils import *
from image_history_buffer import *
from IPython import display
from models import *
from additional_models import *
from collections import deque
from ImageDataGenerator import *
from time import sleep

class _GAN:
    def __init__(self, gen, disc, disc_entropy,DCGAN,GenClass,classificator, batch_size, img_source_dim,
                  img_dest_dim, noise_dim, noise_scale, lr_D, lr_G, deterministic, inject_noise, model, lsmooth,
                 img_buffer, datagen, disc_type, data_aug, n_classes, disc_iters,name, dir ):
        self.generator_model = gen
        self.discriminator_model = disc
        self.discriminator2 = disc_entropy
        self.DCGAN_model = DCGAN
        self.GenClass_model = GenClass
        self.classificator_model = classificator
        self.batch_size = batch_size
        self.img_source_dim = img_source_dim
        self.img_dest_dim = img_dest_dim
        self.bn_mode = 2
        self.noise_dim = noise_dim
        self.noise_scale = noise_scale
        self.lr_D = lr_D
        self.lr_G = lr_G
        self.deterministic = deterministic
        self.inject_noise = inject_noise
        self.img_buffer = img_buffer
        self.datagen = datagen
        self.model = model
        self.lsmooth = lsmooth
        self.disc_type = disc_type
        self.dir=dir
        self.disc_iters=disc_iters
        self.data_aug = data_aug
        self.img_dim = img_dest_dim
        self.n_classes = n_classes
        self.name = name

def load_data(img_dim, image_dim_ordering, dset):
    # Load and normalize data
    if dset == "mnistM":
        X_source_train, Y_source_train, X_source_test, Y_source_test, n_classes1 = load_image_dataset(
            img_dim, image_dim_ordering, dset='mnist')
        X_dest_train, Y_dest_train, X_dest_test, Y_dest_test, n_classes2 = load_image_dataset(
            img_dim, image_dim_ordering, dset='mnistM')
    elif dset == "svhn_to_mnist":
        X_source_train,Y_source_train, X_source_test, Y_source_test, n_classes1 = load_image_dataset(img_dim, image_dim_ordering,dset='svhn')
        X_dest_train,Y_dest_train, X_dest_test, Y_dest_test, n_classes2 = load_image_dataset(img_dim, image_dim_ordering,dset='mnist')
    elif dset == "svhn_to_mnist32":
        X_source_train,Y_source_train, X_source_test, Y_source_test, n_classes1 = load_image_dataset(img_dim, image_dim_ordering,dset='svhn32')
        X_dest_train,Y_dest_train, X_dest_test, Y_dest_test, n_classes2 = load_image_dataset(img_dim, image_dim_ordering,dset='mnist32')
    elif dset == "svhn_to_mnist32gray":
        X_source_train,Y_source_train, X_source_test, Y_source_test, n_classes1 = load_image_dataset(img_dim, image_dim_ordering,dset='svhn32gray')
        X_dest_train,Y_dest_train, X_dest_test, Y_dest_test, n_classes2 = load_image_dataset(img_dim, image_dim_ordering,dset='mnist32')
    elif dset == "mnist_to_svhn32gray":
        X_source_train,Y_source_train, X_source_test, Y_source_test, n_classes1 = load_image_dataset(img_dim, image_dim_ordering,dset='mnist32')
        X_dest_train,Y_dest_train, X_dest_test, Y_dest_test, n_classes2 = load_image_dataset(img_dim, image_dim_ordering,dset='svhn32gray')
    elif dset == "mnist_to_svhn32":
        X_source_train,Y_source_train, X_source_test, Y_source_test, n_classes1 = load_image_dataset(img_dim, image_dim_ordering,dset='mnist32')
        X_dest_train,Y_dest_train, X_dest_test, Y_dest_test, n_classes2 = load_image_dataset(img_dim, image_dim_ordering,dset='svhn32')
    elif dset == "mnist_to_usps":
        X_source_train,Y_source_train, X_source_test, Y_source_test, n_classes1 = load_image_dataset(img_dim, image_dim_ordering,dset='mnist')
        X_dest_train,Y_dest_train, X_dest_test, Y_dest_test, n_classes2 = load_image_dataset(img_dim, image_dim_ordering,dset='usps')
    elif dset == "usps_to_mnist":
        X_source_train,Y_source_train, X_source_test, Y_source_test, n_classes1 = load_image_dataset(img_dim, image_dim_ordering,dset='usps')
        X_dest_train,Y_dest_train, X_dest_test, Y_dest_test, n_classes2 = load_image_dataset(img_dim, image_dim_ordering,dset='mnist')
    elif dset == "MnistMtoMnist":
        X_source_train, Y_source_train, X_source_test, Y_source_test, n_classes1 = load_image_dataset(
            img_dim, image_dim_ordering, dset='mnist')
        X_dest_train, Y_dest_train, X_dest_test, Y_dest_test, n_classes2 = load_image_dataset(
            img_dim, image_dim_ordering, dset='mnistM')
    elif dset == "OfficeAmazonToDslr":
        X_source_train, Y_source_train, X_source_test, Y_source_test, n_classes1 = load_image_dataset(
            img_dim, image_dim_ordering, dset='OfficeAmazon')
        X_dest_train, Y_dest_train, X_dest_test, Y_dest_test, n_classes2 = load_image_dataset(
            img_dim, image_dim_ordering, dset='OfficeDslr')
    elif dset == "OfficeAmazonToWebcam":
        X_source_train, Y_source_train, X_source_test, Y_source_test, n_classes1 = load_image_dataset(
            img_dim, image_dim_ordering, dset='OfficeAmazon')
        X_dest_train, Y_dest_train, X_dest_test, Y_dest_test, n_classes2 = load_image_dataset(
            img_dim, image_dim_ordering, dset='OfficeWebcam')
    else:
        print "dataset not supported"
    if n_classes1 != n_classes2:  # sanity check
        print "number of classes mismatch between source and dest domains"
    n_classes = n_classes1
    img_source_dim = X_source_train.shape[-3:]  # is it backend agnostic?
    img_dest_dim = X_dest_train.shape[-3:]
    if (dset == "mnist_to_usps") or (dset == "usps_to_mnist"):
        X_source=X_source_train
        Y_source=Y_source_train
        X_dest=X_dest_train
        Y_dest=Y_dest_train
    else:
        X_dest = np.concatenate([X_dest_train, X_dest_test], axis=0)
        Y_dest = np.concatenate([Y_dest_train, Y_dest_test], axis=0)
        X_source = np.concatenate([X_source_train, X_source_test], axis=0)
        Y_source = np.concatenate([Y_source_train, Y_source_test], axis=0)
    return X_source, Y_source, X_dest, Y_dest, n_classes, img_source_dim, img_dest_dim

def load_testset(img_dim, image_dim_ordering, dset):
    # Load and normalize data
    if dset == "mnistM":
        X_dest_train, Y_dest_train, X_dest_test, Y_dest_test, n_classes2 = load_image_dataset(
            img_dim, image_dim_ordering, dset='mnistM')
    elif dset == "svhn_to_mnist32":
        X_dest_train,Y_dest_train, X_dest_test, Y_dest_test, n_classes2 = load_image_dataset(img_dim, image_dim_ordering,dset='mnist32')
    elif dset == "svhn_to_mnist32gray":
        X_dest_train,Y_dest_train, X_dest_test, Y_dest_test, n_classes2 = load_image_dataset(img_dim, image_dim_ordering,dset='mnist32')
    elif dset == "mnist_to_svhn32gray":
        X_dest_train,Y_dest_train, X_dest_test, Y_dest_test, n_classes2 = load_image_dataset(img_dim, image_dim_ordering,dset='svhn32gray')
    elif dset == "mnist_to_svhn32":
        X_dest_train,Y_dest_train, X_dest_test, Y_dest_test, n_classes2 = load_image_dataset(img_dim, image_dim_ordering,dset='svhn32')
    elif dset == "OfficeAmazonToDslr":
        X_dest_train, Y_dest_train, X_dest_test, Y_dest_test, n_classes2 = load_image_dataset(img_dim, image_dim_ordering, dset='OfficeDslr')
    elif dset == "OfficeAmazonToWebcam":
        X_dest_train, Y_dest_train, X_dest_test, Y_dest_test, n_classes2 = load_image_dataset(img_dim, image_dim_ordering, dset='OfficeWebcam')
    elif dset == "usps_to_mnist":
        X_dest_train, Y_dest_train, X_dest_test, Y_dest_test, n_classes2 = load_image_dataset(img_dim, image_dim_ordering, dset='mnist')
    elif dset == "mnist_to_usps":
        X_dest_train, Y_dest_train, X_dest_test, Y_dest_test, n_classes2 = load_image_dataset(img_dim, image_dim_ordering, dset='usps')
    else:
        print "dataset not supported in load_testset function!"
    test_data=X_dest_test
    test_labels=Y_dest_test

    return test_data, test_labels

def build_opt(opt_D, opt_G, lr_D, lr_G,lr_rec=None,opt_rec=None):
    _opt_D = get_optimizer(opt_D, lr_D)
    _opt_G = get_optimizer(opt_G, lr_G)
    _opt_C = get_optimizer('SGD', 0.01)
    _opt_Z = get_optimizer('Adam', lr_G)
    if opt_rec is None:
        return _opt_D, _opt_G, _opt_C, _opt_Z
    else:
        _opt_rec = get_optimizer(opt_rec, lr_rec)
        return _opt_D, _opt_G, _opt_C, _opt_Z, _opt_rec

def load_compile_reconstructions(generator_model1, generator_model2,noise_dim,img_source_dim1,img_source_dim2,opt_G, opt_rec, classificator_model2=None):
    rec1 = models.reconstructor(generator_model1, generator_model2, noise_dim, img_source_dim1)
    rec2 = models.reconstructor(generator_model2, generator_model1, noise_dim, img_source_dim2)
    rec1.compile(loss='mse',  optimizer=opt_rec)
    rec2.compile(loss='mse',  optimizer=opt_rec)

    if classificator_model2 is not None:
        models.make_trainable(generator_model1, False) #because generator_model1 is already trained by a classificator in a supervised setting
        models.make_trainable(generator_model2, True)
        models.make_trainable(classificator_model2, True)
        recClass = models.reconstructorClass(generator_model1, generator_model2, classificator_model2, noise_dim, img_source_dim1)
        recClass.compile(loss='categorical_crossentropy',  optimizer=opt_rec)
        return rec1,rec2,recClass
    else:
        return rec1,rec2

def load_compile_models(noise_dim, img_source_dim, img_dest_dim, deterministic, pureGAN, wd, loss1, loss2, disc_type, n_classes, opt_D, opt_G, opt_C, opt_Z,suffix=None,pretrained=False):
    # LOAD MODELS:
    generator_model = models.generator_google_mnistM(
        noise_dim, img_source_dim, img_dest_dim, deterministic, pureGAN, wd,suffix)
    discriminator_model,discriminator2 = models.discriminator_dcgan_doubled(img_dest_dim, wd,n_classes,disc_type)
#    classificator_model = models.classificator_svhn(img_dest_dim, n_classes, wd)
    if pretrained:
        classificator_model = models.resnet50classifier(img_dest_dim, n_classes, wd)  
    else:
        classificator_model = models.classificator_google_mnistM(img_dest_dim, n_classes, wd)
    DCGAN_model = models.DCGAN_naive(generator_model, discriminator_model, noise_dim, img_source_dim)
    GenClass_model = models.DCGAN_naive2(generator_model, classificator_model, noise_dim, img_source_dim)
    if not deterministic:
        zclass_model = z_coerence(generator_model, img_source_dim, bn_mode=2, wd=wd,
                                  inject_noise=False, n_classes=n_classes, noise_dim=noise_dim, model_name="zClass")

    # COMPILE MODELS:
    generator_model.compile(loss=loss1, optimizer=opt_G)
    models.make_trainable(discriminator_model, False)
    models.make_trainable(discriminator2, False)
    models.make_trainable(classificator_model, False)
    if disc_type == "simple_disc":
        DCGAN_model.compile(loss=[loss1], optimizer=opt_G)
        models.make_trainable(discriminator_model, True)
        discriminator_model.compile(loss=[loss1], optimizer=opt_D)
    elif disc_type == "nclass_disc":
        DCGAN_model.compile(loss=loss1, optimizer=opt_G)
        GenClass_model.compile(loss=['categorical_crossentropy'], optimizer=opt_G)
        models.make_trainable(discriminator_model, True)
        models.make_trainable(discriminator2, True)
        discriminator_model.compile(loss=loss1, optimizer=opt_D)
        discriminator2.compile(loss=loss2,  optimizer=opt_D)
    models.make_trainable(classificator_model, True)
    classificator_model.compile(loss=loss2, metrics=['accuracy'], optimizer=opt_C)
    if not deterministic:
        zclass_model.compile(loss=[loss1], optimizer=opt_Z)
        return generator_model, discriminator_model,discriminator2, classificator_model, DCGAN_model,GenClass_model, zclass_model
    else:
        return generator_model, discriminator_model,discriminator2,  classificator_model, DCGAN_model,GenClass_model, None


def load_pretrained_weights(generator_model, discriminator_model,discriminator2, DCGAN_model, name, data, labels, noise_scale, classificator_model=None, resume=False):
    if resume:  # loading previous saved model weights and checking actual performance
        load_model_weights(generator_model, discriminator_model, DCGAN_model, name, classificator_model, discriminator2=discriminator2)
        #loss4, acc4 = classificator_model.evaluate(data, labels, batch_size=512, verbose=0)
        #print('\n Classifier Accuracy on full target domain:  %.2f%%' % (100 * acc4))


def load_buffer_and_augmentation(history_size, batch_size, img_source_dim, n_classes):

    max_history_size = int(history_size * batch_size)
    img_buffer = ImageHistoryBuffer(
        (0,) + img_source_dim, max_history_size, batch_size, n_classes)
    datagen = ImageDataGenerator(rotation_range=0.45,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 fill_mode='nearest')
#    datagen = ImageDataGenerator(elastic_distortion=True)
    return img_buffer, datagen


def get_loss_list():
    list_disc_loss_real = deque(10 * [0], 10)
    list_disc_loss_gen = deque(10 * [0], 10)
    list_gen_loss = deque(10 * [0], 10)
    list_zclass_loss = deque(10 * [0], 10)
    list_classifier_loss = deque(10 * [0], 10)
    list_GenClass_loss = deque(10 * [0], 10)
    l_rec = deque(10 * [0], 10)
    l_recClass = deque(10 * [0], 10)
    return list_disc_loss_real, list_disc_loss_gen, list_gen_loss, list_zclass_loss, list_classifier_loss, l_rec, list_GenClass_loss, l_recClass

def get_batch(A_data, A_labels, B_data, B_labels, batch_size):
    A_data_batch, A_labels_batch, _ = next(gen_batch(A_data, A_labels, batch_size))
    B_data_batch, B_labels_batch, _ = next(gen_batch(B_data, B_labels, batch_size))
    return A_data_batch, A_labels_batch, B_data_batch, B_labels_batch

def train_gan(GAN, disc_iters, A_data, A_labels, B_data, B_labels, batch_counter, l_disc_real, l_disc_gen, l_gen,l_GenClass,class_weight):
    if GAN.dir == 'BtoA':
        for disc_it in range(disc_iters):
            A_data_batch, A_labels_batch, B_data_batch, B_labels_batch = get_batch(A_data, A_labels, B_data, B_labels, GAN.batch_size)
            X_source_batch = B_data_batch
            #Y_source_batch = B_labels_batch
            X_dest_batch = A_data_batch
            Y_dest_batch = A_labels_batch

            ##########
            # Create a batch to feed the discriminator model
            #########
            X_noise = sample_noise(GAN.noise_scale, GAN.batch_size, GAN.noise_dim)
            gen_output = GAN.generator_model.predict([X_noise,X_source_batch])
            #X_disc_real, X_disc_gen = get_disc_batch(X_dest_batch, GAN.generator_model, batch_counter, GAN.batch_size,
            #                                                    GAN.noise_dim, X_source_batch, noise_scale=GAN.noise_scale)
            if GAN.disc_type == "simple_disc":
                current_labels_real = np.ones(GAN.batch_size)
                current_labels_gen = np.zeros(GAN.batch_size)
            if GAN.disc_type == ("nclass_disc"):
                current_labels_real = np.ones(GAN.batch_size) 
                current_labels_gen = np.zeros(GAN.batch_size) 
            ##############
            # Train the disc on gen-buffered samples and on current real samples
            ##############
            disc_loss_real = GAN.discriminator_model.train_on_batch(X_dest_batch, current_labels_real)
            GAN.img_buffer.add_to_buffer(gen_output, current_labels_gen, GAN.batch_size)
            bufferImages, bufferLabels = GAN.img_buffer.get_from_buffer(GAN.batch_size)
            disc_loss_gen = GAN.discriminator_model.train_on_batch(bufferImages, bufferLabels)
            disc2_loss = GAN.discriminator2.train_on_batch(X_dest_batch,Y_dest_batch * 1.0) #GAN.lsmooth) #training the discriminator_classifier model

            l_disc_real.appendleft(disc_loss_real)
            l_disc_gen.appendleft(disc_loss_gen)
        #Train the GENERATOR, it is the same on both AtoB and BtoA:
        X_noise = sample_noise(GAN.noise_scale, GAN.batch_size, GAN.noise_dim)
        if GAN.disc_type == "simple_disc":                
            gen_loss =  GAN.DCGAN_model.train_on_batch([X_noise,X_source_batch], np.ones(GAN.batch_size))
        elif GAN.disc_type == "nclass_disc":
            gen_loss =  GAN.DCGAN_model.train_on_batch([X_noise,X_source_batch], np.ones(GAN.batch_size)) 
            class_p = GAN.GenClass_model.predict([X_noise,X_source_batch]) #AUTOLABELLING
            idx = np.argmax(class_p, axis=1) #
            virtual_labels = (idx[:, None]) == np.arange(GAN.n_classes) * 1.0 #
            GenClass_loss =  GAN.GenClass_model.train_on_batch([X_noise,X_source_batch], virtual_labels) #
        l_GenClass.appendleft(GenClass_loss) #
        l_gen.appendleft(gen_loss)

    elif GAN.dir == 'AtoB':
        for disc_it in range(disc_iters):
            A_data_batch, A_labels_batch, B_data_batch, B_labels_batch = get_batch(A_data, A_labels, B_data, B_labels, GAN.batch_size)
            X_source_batch = A_data_batch
            Y_source_batch = A_labels_batch
            X_dest_batch = B_data_batch
            #Y_dest_batch = B_labels_batch
            X_noise = sample_noise(GAN.noise_scale, GAN.batch_size, GAN.noise_dim)
            gen_output = GAN.generator_model.predict([X_noise,X_source_batch])
            #X_disc_real, X_disc_gen = get_disc_batch(X_dest_batch, GAN.generator_model, batch_counter, GAN.batch_size,
            #                                                    GAN.noise_dim, X_source_batch, noise_scale=GAN.noise_scale)
            if GAN.disc_type == "simple_disc":
                current_labels_real = np.ones(GAN.batch_size)
                current_labels_gen = np.zeros(GAN.batch_size)
            if GAN.disc_type == ("nclass_disc"):
                current_labels_real = np.ones(GAN.batch_size) 
                current_labels_gen = np.zeros(GAN.batch_size) 
            ##############
            #Train the disc on gen-buffered samples and on current real samples
            ##############
            disc_loss_real = GAN.discriminator_model.train_on_batch(X_dest_batch, current_labels_real)
            GAN.img_buffer.add_to_buffer(gen_output,current_labels_gen, GAN.batch_size)
            bufferImages, bufferLabels = GAN.img_buffer.get_from_buffer(GAN.batch_size)
            disc_loss_gen = GAN.discriminator_model.train_on_batch(bufferImages, bufferLabels)
            disc2_loss = GAN.discriminator2.train_on_batch(gen_output,Y_source_batch * 1.0) #GAN.lsmooth) #training the discriminator_classifier model

            l_disc_real.appendleft(disc_loss_real)
            l_disc_gen.appendleft(disc_loss_gen)

        #Train the GENERATOR, it is the same on both AtoB and BtoA:
        X_noise = sample_noise(GAN.noise_scale, GAN.batch_size, GAN.noise_dim)
        if GAN.disc_type == "simple_disc":                
            gen_loss =  GAN.DCGAN_model.train_on_batch([X_noise,X_source_batch], np.ones(GAN.batch_size)) #TRYING SAME BATCH OF DISC
        elif GAN.disc_type == "nclass_disc":
            gen_loss =  GAN.DCGAN_model.train_on_batch([X_noise,X_source_batch], np.ones(GAN.batch_size)) #TRYING SAME BATCH OF DISC
            GenClass_loss =  GAN.GenClass_model.train_on_batch([X_noise,X_source_batch], Y_source_batch,sample_weight=np.ones(GAN.batch_size)*class_weight)
            #gen_loss = gen_loss[0]
        l_gen.appendleft(gen_loss)
        l_GenClass.appendleft(GenClass_loss)
    return A_data_batch, A_labels_batch, B_data_batch, B_labels_batch

def train_class(GAN, l_class,  A_data_batch, A_labels_batch):
    if GAN.dir == 'AtoB':
        X_noise = sample_noise(GAN.noise_scale, GAN.batch_size, GAN.noise_dim)
        if GAN.data_aug:
            x_dest_batch = GAN.generator_model.predict([X_noise,datagen.output(A_data_batch)])
        else:
            x_dest_batch = GAN.generator_model.predict([X_noise,A_data_batch])
        # NO LABEL SMOOTHING!!!! inverted training w.r.t. to AtoB, because I
        # have labels of A
        class_loss = GAN.classificator_model.train_on_batch(x_dest_batch, A_labels_batch)
    elif GAN.dir == 'BtoA':
        class_loss = GAN.classificator_model.train_on_batch(A_data_batch, A_labels_batch)
    l_class.appendleft(class_loss[0])
    return l_class

def train_rec(GAN,rec1, rec2, A_data_batch, B_data_batch, l_rec1, l_rec2,rec_weight):
    X_noise = sample_noise(GAN.noise_scale, GAN.batch_size, GAN.noise_dim)
    X_noise2 = sample_noise(GAN.noise_scale, GAN.batch_size, GAN.noise_dim)
    rec_loss = rec1.train_on_batch([X_noise, A_data_batch,X_noise2],A_data_batch,sample_weight=np.ones(GAN.batch_size)*rec_weight)
    rec_loss2 = rec2.train_on_batch([X_noise, B_data_batch,X_noise2],B_data_batch,sample_weight=np.ones(GAN.batch_size)*rec_weight)
    l_rec1.appendleft(rec_loss)
    l_rec2.appendleft(rec_loss2)
    return l_rec1, l_rec2

def train_recClass(GAN,recClass, A_data_batch, A_labels_batch,  l_recClass, rec_weight):
    X_noise = sample_noise(GAN.noise_scale, GAN.batch_size, GAN.noise_dim)
    X_noise2 = sample_noise(GAN.noise_scale, GAN.batch_size, GAN.noise_dim)
    recClass_loss = recClass.train_on_batch([X_noise, A_data_batch,X_noise2],A_labels_batch,sample_weight=np.ones(GAN.batch_size)*rec_weight)
    l_recClass.appendleft(recClass_loss)
    return l_recClass


def train_gen_zclass(generator_model, DCGAN_model, zclass_model, disc_type, deterministic, noise_dim, noise_scale, batch_size, l_gen, l_zclass, X_source, Y_source, n_classes):
    X_gen = sample_noise(noise_scale, batch_size, noise_dim)
    X_source_batch2, Y_source_batch2, idx_source_batch2 = next(
        gen_batch(X_source, Y_source, batch_size))
    if disc_type == "simple_disc":
        gen_loss = DCGAN_model.train_on_batch([X_gen, X_source_batch2], np.ones(X_gen.shape[0]))  # TRYING SAME BATCH OF DISC
    elif disc_type == ("nclass_disc"):

        #(disc_p, class_p) = DCGAN_model.predict_on_batch(X_source_batch2)
        #idx = np.argmax(class_p, axis=1)
        #virtual_labels = (idx[:, None] == np.arange(n_classes)) * 1

        virtual_labels = np.zeros([GAN.batch_size, GAN.n_classes])
        gen_loss = DCGAN_model.train_on_batch([X_gen, X_source_batch2], [np.ones(X_gen.shape[0]), virtual_labels])  # FIX :((
        #gen_loss = gen_loss[0]
    l_gen.appendleft(gen_loss)
    if not deterministic:
        zclass_loss = zclass_model.train_on_batch(
            [X_gen, X_source_batch2], [X_gen])
    else:
        zclass_loss = 0.0
    l_zclass.appendleft(zclass_loss)
    return l_gen, l_zclass


def visualize_save_stuffs(GANs, progbar, gen_iterations, batch_counter, n_batch_per_epoch, l_disc_real1, l_disc_gen1, l_gen_loss1,
                          l_class_loss1, l_disc_real2, l_disc_gen2, l_gen_loss2,l_class_loss2, A_data, A_labels, B_data, B_labels,
                          start,e, l_rec1, l_rec2,l_GenClass1, l_GenClass2, l_recClass):
    gen_iterations += 1
    batch_counter += 1
    image_dim_ordering = 'th'
    progbar.add(GANs[0].batch_size, values=[("Loss_D_real1", np.mean(l_disc_real1)),
                                            ("Loss_D_gen1", np.mean(l_disc_gen1)),
                                            ("Loss_G1", np.mean(l_gen_loss1)),
                                            ("Loss_Classifier1",np.mean(l_class_loss1)),
                                            ("Loss_D_real2", np.mean(l_disc_real2)),
                                            ("Loss_D_gen2", np.mean(l_disc_gen2)),
                                            ("Loss_G2", np.mean(l_gen_loss2)),
                                            ("Loss_Classifier2",np.mean(l_class_loss2)),
                                            ("Loss_Rec1", np.mean(l_rec1)),
                                            ("Loss_Rec2", np.mean(l_rec2)),
                                            ("Loss_GenClass1", np.mean(l_GenClass1)),
                                            ("Loss_AutoLabel", np.mean(l_GenClass2)),
                                            ("Loss_RecClass", np.mean(l_recClass))
                                       ])

    for GAN in GANs:
        # plot images 1 times per epochs        if GAN.dir == 'BtoA':
        if GAN.dir == 'BtoA':
            X_source=B_data
            Y_source=B_labels
            X_dest = A_data
            Y_dest = A_labels
        elif GAN.dir == 'AtoB':
            X_source=A_data
            Y_source=A_labels
            X_dest = B_data
            Y_dest = B_labels

        if batch_counter == n_batch_per_epoch:
        #if batch_counter % (n_batch_per_epoch) == 0:
            X_source_batch_plot, Y_source_batch_plot, idx_source_plot = next(gen_batch(X_source, Y_source, batch_size=GAN.batch_size))
            returned_idx = plot_generated_batch(X_dest, X_source, GAN.generator_model, GAN.noise_dim, image_dim_ordering, idx_source_plot,
                                        batch_size=GAN.batch_size,different_idx=True, datagen=GAN.datagen, data_aug=GAN.data_aug)
            print ("Dest labels:") 
            print (Y_dest[returned_idx].argmax(1))
            print ("Source labels:") 
            print (Y_source_batch_plot.argmax(1))
            print('\nEpoch %s, Time: %s' % (e + 1, time.time() - start))
        else:
            idx_source_plot = 0
            Y_source_batch_plot = 0

         #Save model weights (by default, every 5 epochs)
        if batch_counter == n_batch_per_epoch:
            save_model_weights(GAN.generator_model, GAN.discriminator_model,
                                      GAN.DCGAN_model, e, GAN.name, GAN.classificator_model, discriminator2=GAN.discriminator2)
    return batch_counter, gen_iterations

def pretrain_disc( GAN, A_data, A_labels,B_data, B_labels,class_weight, pretrain_iters=100, resume=False):
    l_real = deque(10 * [0], 10)
    l_gen = deque(10 * [0], 10)
    l_genclass = deque(10 * [0], 10)
    if not resume:
        _, _, _, _ = train_gan(GAN, pretrain_iters, A_data, A_labels, B_data, B_labels, 1, l_real, l_gen,l_gen,l_genclass,class_weight)
        print "Pretrain of discriminator finished."
    else:
        print "resumed previous training."


def testing_class_accuracy(GANs,classificator_model, generator_model, vis_samples, noise_dim, noise_scale, data, labels):
    acc=[]
    loss=[]
    for GAN in GANs:
        if GAN.dir == 'BtoA':
            # testing accuracy of trained classifier
            X_noise = sample_noise(GAN.noise_scale, vis_samples, GAN.noise_dim)
            Xsource_dataset_mapped = GAN.generator_model.predict(
                [X_noise, data[:vis_samples]], batch_size=1000)
            true_labels = labels[:vis_samples]
            p1 = GAN.classificator_model.predict(Xsource_dataset_mapped, batch_size=1000, verbose=1)
            score1 = np.sum(np.argmax(true_labels,axis=1) == np.argmax(p1, axis=1)) / float(true_labels.shape[0]) 
            print('\n Classifier Accuracy and loss on full target domain:  %.2f%%  ' %
                  ((100 * score1)))

        if GAN.dir == 'AtoB':
            X_noise = sample_noise(GAN.noise_scale, vis_samples, GAN.noise_dim)
            Xsource_dataset_mapped = data[:vis_samples]
            true_labels = labels[:vis_samples]
            p2 = GAN.classificator_model.predict(Xsource_dataset_mapped, batch_size=1000, verbose=1)
            score2 = np.sum(np.argmax(true_labels,axis=1) == np.argmax(p2, axis=1)) / float(true_labels.shape[0]) 
            print('\n Classifier Accuracy and loss on full target domain:  %.2f%%  ' %
                  ((100 * score2)))
    
    res = []
    for x in np.arange(0, 1.1, 0.1):
        res.append((x, np.sum(np.argmax(true_labels,axis=1) == np.argmax(p1*x + p2*(1-x), axis=1)) / float(true_labels.shape[0])))
    for (x, score) in res:
        print("\n Coeff: %f - score: %.2f" % (x, score*100))



def train(**kwargs):
    """
    Train standard DCGAN model

    args: **kwargs (dict) keyword arguments that specify the model hyperparameters
    """

    # Roll out the parameters
    generator = kwargs["generator"]
    discriminator = kwargs["discriminator"]
    dset = kwargs["dset"]
    img_dim = kwargs["img_dim"]
    nb_epoch = kwargs["nb_epoch"]
    batch_size = kwargs["batch_size"]
    n_batch_per_epoch = kwargs["n_batch_per_epoch"]
    bn_mode = kwargs["bn_mode"]
    noise_dim = kwargs["noise_dim"]
    noise_scale = kwargs["noise_scale"]
    lr_rec = kwargs["lr_D"]    
    opt_rec = kwargs["opt_rec"]
    lr_G = kwargs["lr_G"]
    lr_D = kwargs["lr_D"]
    opt_D = kwargs["opt_D"]
    opt_G = kwargs["opt_G"]
    use_mbd = kwargs["use_mbd"]
    image_dim_ordering = kwargs["image_dim_ordering"]
    epoch_size = n_batch_per_epoch * batch_size
    deterministic1 = kwargs["deterministic1"]
    deterministic2 = kwargs["deterministic2"]
    inject_noise = kwargs["inject_noise"]
    model = kwargs["model"]
    no_supertrain = kwargs["no_supertrain"]
    pureGAN = kwargs["pureGAN"]
    lsmooth = kwargs["lsmooth"]
    disc_type = kwargs["disc_type"]
    resume = kwargs["resume"]
    name = kwargs["name"]
    wd = kwargs["wd"]
    history_size = kwargs["history_size"]
    monsterClass = kwargs["monsterClass"]
    data_aug = kwargs["data_aug"]
    disc_iters = kwargs["disc_iterations"]
    class_weight = kwargs["class_weight"]
    reconst_w= kwargs["reconst_w"]
    rec = kwargs["rec"]
    reconstClass = kwargs["reconstClass"]
    pretrained = kwargs["pretrained"]
    print("\nExperiment parameters:")
    for key in kwargs.keys():
        print key, kwargs[key]
    print("\n")
    #####some extra parameters:
    
    noise_dim = (noise_dim,)
    name1 = name + '1'
    name2 = name + '2'
    # Setup environment (logging directory etc)
    general_utils.setup_logging("DCGAN")
    gen_iterations = 0
    # Loading data
    A_data, A_labels, B_data, B_labels, n_classes, img_A_dim, img_B_dim = load_data(img_dim, image_dim_ordering, dset)
    test_data, test_labels  = load_testset(img_dim, image_dim_ordering, dset)
    if deterministic1 is None:
        deterministic1 = False   
    if deterministic2 is None:
        deterministic2 = False

    opt_D1, opt_G1, opt_C1, opt_Z1, opt_rec = build_opt(opt_D, opt_G, lr_D, lr_G, lr_rec, opt_rec)
    generator_model1, discriminator_model1,discriminator_class1, classificator_model1, DCGAN_model1, GenClass_model1, zclass_model1 = load_compile_models(noise_dim, img_A_dim, img_B_dim,
                                                     deterministic1, pureGAN, wd, 'mse', 'categorical_crossentropy', disc_type, n_classes, opt_D1, opt_G1, opt_C1, opt_Z1, suffix=None, pretrained=pretrained)
    load_pretrained_weights(generator_model1, discriminator_model1,discriminator_class1, DCGAN_model1, name1, B_data, B_labels, noise_scale, classificator_model1, resume=resume)
    img_buffer1, datagen1 = load_buffer_and_augmentation(history_size, batch_size, img_A_dim, n_classes)

    GAN1=_GAN(generator_model1, discriminator_model1, discriminator_class1,DCGAN_model1,GenClass_model1,classificator_model1, batch_size, img_A_dim,img_B_dim, noise_dim, noise_scale,
               lr_D, lr_G, deterministic1, inject_noise, model, lsmooth, img_buffer1, datagen1, disc_type, data_aug, n_classes, disc_iters,name1, dir='AtoB' )
    pretrain_disc( GAN1, A_data,A_labels, B_data, B_labels,class_weight, pretrain_iters=500, resume=resume)
    #####################

   ##### Setup GAN2
    opt_D2, opt_G2, opt_C2, opt_Z2 = build_opt(opt_D, opt_G, lr_D, lr_G)
    generator_model2, discriminator_model2, discriminator_class2, classificator_model2, DCGAN_model2, GenClass_model2, zclass_model2 = load_compile_models(noise_dim, img_B_dim, img_A_dim,
                                                       deterministic2, pureGAN, wd, 'mse', 'categorical_crossentropy', disc_type, n_classes, opt_D2, opt_G2, opt_C2, opt_Z2, suffix=True)
    load_pretrained_weights(generator_model2, discriminator_model2,discriminator_class2, DCGAN_model2, name2, B_data, B_labels, noise_scale, classificator_model2, resume=resume)
    img_buffer2, datagen2 = load_buffer_and_augmentation(history_size, batch_size, img_B_dim, n_classes)

    gen_entropy2=None
    GAN2=_GAN(generator_model2, discriminator_model2, discriminator_class2, DCGAN_model2,GenClass_model2,classificator_model2, batch_size, img_B_dim,img_A_dim, noise_dim, noise_scale,
               lr_D, lr_G, deterministic2, inject_noise, model, lsmooth, img_buffer2, datagen2, disc_type, data_aug, n_classes, disc_iters, name2, dir='BtoA' )
    pretrain_disc( GAN2, A_data,A_labels, B_data, B_labels,class_weight, pretrain_iters=500, resume=resume)
    ################

    ####Reconstruction losses between Gen1 and Gen2:
    rec1, rec2, recClass = load_compile_reconstructions(generator_model1, generator_model2,noise_dim,img_A_dim,img_B_dim,opt_G,opt_rec, classificator_model2)  

    if resume:
        testing_class_accuracy([GAN1,GAN2],GAN1.classificator_model, GAN1.generator_model,
                               test_data.shape[0], GAN1.noise_dim, GAN1.noise_scale, test_data, test_labels)

#    if two_gans:
#        GANs=[GAN1,GAN2]
#    else:
#        GANs=[GAN1]
    ################
    ##################
    for e in range(1, nb_epoch + 1):
        # Initialize progbar and batch counter
        progbar = generic_utils.Progbar(epoch_size,interval=0.2)
        batch_counter = 1
        start = time.time()
        while batch_counter < n_batch_per_epoch:
            l_disc_real1, l_disc_gen1, l_gen1, l_z1, l_class1,l_rec1,l_GenClass1,_ = get_loss_list()
            A_data_batch, A_labels_batch, B_data_batch, B_labels_batch = train_gan(GAN1, GAN1.disc_iters, A_data, A_labels, B_data, B_labels, batch_counter, l_disc_real1, l_disc_gen1, l_gen1,l_GenClass1, class_weight)

            l_disc_real2, l_disc_gen2, l_gen2, l_z2, l_class2, l_rec2, l_GenClass2, l_recClass = get_loss_list()
            A_data_batch, A_labels_batch, B_data_batch, B_labels_batch = train_gan(GAN2, GAN2.disc_iters, A_data, A_labels, B_data, B_labels, batch_counter, l_disc_real2, l_disc_gen2,l_gen2,l_GenClass2, class_weight)
            if rec:
                train_rec(GAN1, rec1, rec2, A_data_batch, B_data_batch,l_rec1, l_rec2,reconst_w) #BRINGING US TO L.A.? :)
            if reconstClass > 0.0:
                train_recClass(GAN1,recClass, A_data_batch, A_labels_batch,  l_recClass, reconstClass)
            l_class1 = train_class(GAN1, l_class1,  A_data_batch, A_labels_batch)
            l_class2 = train_class(GAN2, l_class2,  A_data_batch, A_labels_batch)

            batch_counter, gen_iterations = visualize_save_stuffs([GAN1,GAN2], progbar, gen_iterations, batch_counter, n_batch_per_epoch, 
                                                                              l_disc_real1, l_disc_gen1, l_gen1, l_class1, l_disc_real2, l_disc_gen2,
                                                                              l_gen2, l_class2, A_data, A_labels, B_data, B_labels,start,e,l_rec1, l_rec2,
                                                                              l_GenClass1, l_GenClass2, l_recClass)


        testing_class_accuracy([GAN1,GAN2],GAN1.classificator_model, GAN1.generator_model,
                               test_data.shape[0], GAN1.noise_dim, GAN1.noise_scale, test_data, test_labels)
#        testing_class_accuracy([GAN1],GAN1.classificator_model, GAN1.generator_model,
#                               5000, GAN1.noise_dim, GAN1.noise_scale, B_data, B_labels)

