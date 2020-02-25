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


defaultParams = {

    'no_classes': 5,                                   # Number of classes in the N-way K-shot learning case
    'no_shots': 1,                                     # Number of 'shots' in the few-shots learning
    'rand_seed':0,                                     # Select the random seed file for taking the weights
    'no_filters' : 64,                                 # Numebr of filters in the convolutional layers
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


    def send_inputs(self):
        input_val = input_generator()
        inputdata = input_val.dataset_reader(self.params['address'])
        #print (self.params['steps'])
        inputs, labels, testlabel  = input_val.gen_inputs_labels_testlabel(self.params, inputdata, test=False)

params = {}
params.update(defaultParams)
print (params)
emulate = omniglot_hd_emulation(params)
emulate.send_inputs()
