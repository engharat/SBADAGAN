"""
Module implementing the image history buffer described in `2.3. Updating Discriminator using a History of
Refined Images` of https://arxiv.org/pdf/1612.07828v1.pdf.

"""
import code
import numpy as np
from sklearn.utils import shuffle

class ImageHistoryBuffer(object):
    def __init__(self, shape, max_size, batch_size, n_classes):
        """
        Initialize the class's state.

        :param shape: Shape of the data to be stored in the image history buffer
                      (i.e. (0, img_height, img_width, img_channels)).
        :param max_size: Maximum number of images that can be stored in the image history buffer.
        :param batch_size: Batch size used to train GAN.
        """
#        code.interact(local=locals())
        self.buffer = None #np.zeros(1) #np.zeros(shape=shape)
        self.labels = None #np.zeros(1) #np.zeros(shape=(shape[0],n_classes)) ##this will be overwritten by add to buffer function
        self.max_size = max_size
        self.batch_size = batch_size
        self.first_time = True

    def add_to_buffer(self, images,labels, nb_to_add=None):
        """
        To be called during training of GAN. By default add batch_size // 2 images to the image history buffer each
        time the generator generates a new batch of images.

        :param images: Array of images (usually a batch) to be added to the image history buffer.
        :param nb_to_add: The number of images from `images` to add to the image history buffer
                          (batch_size / 2 by default).
        """
        if not nb_to_add:
            nb_to_add = self.batch_size // 2

        if self.first_time:
            self.first_time = False
            self.buffer = np.copy(images)
            #code.interact(local=locals())
            if len(labels)== 2:
                self.labels = np.copy(labels[0])	
                self.labels1 = np.copy(labels[1])
                self.multilabels = True	
            else:
                self.labels = np.copy(labels)
            	self.multilabels = False
            #np.append(self.labels, labels[:self.batch_size], axis=0)
        elif (len(self.buffer) + nb_to_add) <= self.max_size: # ex: 10 + 32 < 50 so I don't have to remove anything
            #code.interact(local=locals())
            np.append(self.buffer, images[:nb_to_add], axis=0)
            if self.multilabels:    
                np.append(self.labels, labels[0][:nb_to_add], axis=0)
                np.append(self.labels1, labels[1][:nb_to_add], axis=0)
            else:
                np.append(self.labels, labels[:nb_to_add], axis=0)
        elif (len(self.buffer) + nb_to_add ) > self.max_size:
            #code.interact(local=locals())
            self.buffer[:nb_to_add] = images[:nb_to_add]
            if self.multilabels:    
                self.labels[:nb_to_add] = labels[0][:nb_to_add]
                self.labels1[:nb_to_add] = labels[1][:nb_to_add]
            else:
                self.labels[:nb_to_add] = labels[:nb_to_add]
        else:
            #code.interact(local=locals())
            assert False
        #code.interact(local=locals())
       # self.buffer, self.labels = shuffle(self.buffer, self.labels)

        
    def get_from_buffer(self, nb_to_get=None):
        """
        Get a random sample of images from the history buffer.

        :param nb_to_get: Number of images to get from the image history buffer (batch_size / 2 by default).
        :return: A random sample of `nb_to_get` images from the image history buffer, or an empty np array if the image
                 history buffer is empty.
        """
        if not nb_to_get:
            nb_to_get = self.batch_size // 2

        try:
            if self.multilabels:    
                returned_labels = [self.labels[:nb_to_get],self.labels1[:nb_to_get]]
            else:
                returned_labels = self.labels[:nb_to_get]
            return self.buffer[:nb_to_get], returned_labels
        except IndexError:
            return np.zeros(shape=0),np.zeros(shape=0)
