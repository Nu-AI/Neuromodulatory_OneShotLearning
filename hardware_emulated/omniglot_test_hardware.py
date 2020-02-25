import numpy as np
from numpy import random
import torch.nn.functional as F
from torch import optim
import random
import sys
import pickle
import pdb
import time
import skimage
from skimage import transform
import os
import platform
from conv import *
from weight_loader import *
from input_generator import *


image = skimage.data.chelsea()
img  = skimage.color.rgb2gray(image)
l1_filter = np.zeros((2,3,3))
l1_filter_new = np.zeros((2,3,3))
img = skimage.transform.resize(img, (31,31))
img = img/np.linalg.norm(img)
norm_image = map(lambda row:row/np.linalg.norm(row), img)


l1_filter[0, :, :] = np.array([[[-1, 0, 1],
                                  [1, 0, 1],
                                  [-1, 0, 1]]])

l1_filter[1, :, :] = np.array([[[1,   1,  -1],
                                [0,   0,  0],
                               [-1, 1, 1]]])

l1_filter_new[0, :, :] = np.array([[[1, -1, 1],
                                  [0, 0, 1],
                                  [1, 0, -1]]])

l1_filter_new[1, :, :] = np.array([[[1,   0,  -1],
                                [0,   1,  0],
                               [-1, 0, 1]]])

l2_filter =  np.zeros((2,2,3,3))
l2_filter[0,:,:,:] = l1_filter
l2_filter[1,:,:,:] = l1_filter_new




defaultParams = {

    'no_classes': 5,                                   # Number of classes in the N-way K-shot learning case
    'no_shots': 1,                                     # Number of 'shots' in the few-shots learning
    'rand_seed':0,                                     # Select the random seed file for taking the weights
    'no_filters' : 64,                                 # Number of filters in the convolutional layers
    'imagesize': 31,                                   # The size of the 2D images to be reshaped to
    'learningrate': 1e-5,                              # The initial learning rate for the network
    'present_test': 1,                                 # Number of times we present the testing class
    'address': '../omniglot_dataset/omniglot/python/', # enter the path of the dataset here
    #'print_every': 10,  # After how many epochs
}
TEST_CLASSES = 100

class omniglot_hd_emulation:

    def __init__(self, params):
        self.params = params
        self.params['steps'] =  params['no_shots']*(params['no_classes']) + params['present_test']
        print ("Started the emulation")


    def read_inputs(self):
        input_val = input_generator()
        inputdata = input_val.dataset_reader(self.params['address'])
        inputs, labels, testlabel  = input_val.gen_inputs_labels_testlabel(self.params, inputdata, test=False)


    def read_weights(self):
        load_weights = weight_loader()
        dict1, tmpw, tmpalpha, tmpeta =weight_load.read_fc_params(1)
        print ("Done loading weights")

    def Network(self, img):
        conv1 = conv_3_3(img, 3,2,2)
        #feature_maps =
        conv2  = conv_3_3(conv1.forward(l1_filter), 3,2,2)
        #new_feature_maps= conv2.forward(l2_filter)
        #print(new_feature_maps,new_feature_maps.shape)
        conv3  = conv_3_3(conv2.forward(l2_filter), 3,2,2)
        #updated_f_maps = conv3.forward(l2_filter)
        #print (updated_f_maps,updated_f_maps.shape)
        conv4 = conv_3_3(conv3.forward(l2_filter),3,2,2)
        #updated_final_maps = conv4.forward(l2_filter)
        print (conv4.forward(l2_filter), "This is the final conv layer output")

params = {}
params.update(defaultParams)
print (params)
emulate = omniglot_hd_emulation(params)
emulate.read_inputs()
emulate.read_weights()
emulate.Network(img)
