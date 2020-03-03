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
import seaborn as sns
import matplotlib.pyplot as plt
from conv import *
from weight_loader import *
from input_generator import *

from conv_fp32 import *
import ctypes, ctypes.util
from ctypes import cdll

sns.set()
##################################################################
# The next section involves setting up the C++ wrappers for the  #
# given convolutional operations in fixed point.                 #
##################################################################
# Fixedlib = ctypes.CDLL("FixedPoint.so")
#
# Fixedlib.Float_to_Fixed.restype = ctypes.c_int32
# Fixedlib.Float_to_Fixed.argtypes = (ctypes.c_float, ctypes.c_int, ctypes.c_int)
#
# Fixedlib.Fixed_to_Float.restype = ctypes.c_float
# Fixedlib.Fixed_to_Float.argtypes = (ctypes.c_float, ctypes.c_int)
#
# Fixedlib.Fixed_to_Float2.restype = ctypes.c_float
# Fixedlib.Fixed_to_Float2.argtypes = (ctypes.c_float, ctypes.c_int)
#
# Fixedlib.Fixed_Mul.restype = ctypes.c_int32
# Fixedlib.Fixed_Mul.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_int)
#
# Fixedlib.Fixed_ACC.restype = ctypes.c_float
# Fixedlib.Fixed_ACC.argtypes = (np.ctypeslib.ndpointer(dtype=np.float32), ctypes.c_int)
#
# def Float_to_Fixed(number, integer, fraction):
#     result = Fixedlib.Float_to_Fixed(number, integer, fraction)
#     return result
#
# def Fixed_to_Float(number, fraction):
#     result = Fixedlib.Fixed_to_Float(number, fraction)
#     return result
#
# def Fixed_to_Float2(number, fraction):
#     result = Fixedlib.Fixed_to_Float2(number, fraction)
#     return result
#
# def Fixed_Mul(input1, input2, integer, fraction):
#     result = Fixedlib.Fixed_Mul(input1, input2, integer, fraction)
#     return result
#
# def Fixed_ACC(Product, shape):
#     result = Fixedlib.Fixed_ACC(Product, shape)
#     return result


#################################################################
# Sample stuff for debugging and testing the functionality      #
#################################################################
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
    'no_test_iters': 1,
    'address': '../omniglot_dataset/omniglot/python/', # enter the path of the dataset here
    #'print_every': 10,  # After how many epochs
}
TEST_CLASSES = 100

class omniglot_hd_emulation:

    def __init__(self, params):
        self.params = params
        self.params['steps'] =  params['no_shots']*(params['no_classes']) + params['present_test']
        print ("Started the emulation")


    def read_input_dataset(self):
        input_val = input_generator()
        inputdata = input_val.dataset_reader(self.params['address'])
        print (np.array(inputdata).shape,"the shape of the input data")
        #inputs, labels, testlabel  = input_val.gen_inputs_labels_testlabel(self.params, inputdata, test=False)
        return inputdata

    def read_inputs(self, dataset):
        input_val = input_generator()
        return input_val.gen_inputs_labels_testlabel(self.params, dataset, test=False)

    def read_weights(self):
        load_weights = weight_loader()
        dict1, tmpw, tmpalpha, tmpeta =weight_load.read_fc_params(1)
        print ("Done loading weights")
        return dict1, tmpw, tmpalpha, tmpeta

    def Network(self, img, kernel_size, stride,dict):
        conv1 = conv_3_3(img, kernel_size,stride,self.params['no_filters'],dict['cv1.bias'])
        #print (conv1.forward(dict['cv1.weight']))
        conv2 = conv_3_3(conv1.forward(dict['cv1.weight']), kernel_size,stride,self.params['no_filters'], dict['cv2.bias'])
        conv3 = conv_3_3(conv2.forward(dict['cv2.weight']), kernel_size,stride,self.params['no_filters'], dict['cv3.bias'])
        conv4 = conv_3_3(conv3.forward(dict['cv3.weight']), kernel_size,stride,self.params['no_filters'], dict['cv4.bias'])

        print (conv4.forward(dict['cv4.weight']), "This is the final conv layer output")
        return conv4.forward(dict['cv4.weight'])

    def Network_fp(self, img, kernel_size, stride, dict):
        conv1 = conv_3_3_fp(img, kernel_size,stride,self.params['no_filters'],dict['cv1.bias'])
        #print (conv1.forward(dict['cv1.weight']))
        conv2 = conv_3_3_fp(conv1.forward(dict['cv1.weight']), kernel_size,stride,self.params['no_filters'], dict['cv2.bias'])
        conv3 = conv_3_3_fp(conv2.forward(dict['cv2.weight']), kernel_size,stride,self.params['no_filters'], dict['cv3.bias'])
        conv4 = conv_3_3_fp(conv3.forward(dict['cv3.weight']), kernel_size,stride,self.params['no_filters'], dict['cv4.bias'])

        print (conv4.forward(dict['cv4.weight']), "This is the final conv layer output")
        return conv4.forward(dict['cv4.weight'])

    def inputs_to_fixed(self, inputs):
        input_fixed_arr = np.empty_like(inputs)
        for i in range(self.params['steps']):
            input_list = list(np.reshape(inputs[i,:,:],(self.params['imagesize']*self.params['imagesize'])))
            #print (temp2_array.shape)
            #input_list = temp2_array.tolist()
            input_fixed = list(map(lambda x: Float_to_Fixed(x,2,12), input_list))
            input_fixed = np.reshape(np.array(input_fixed),(self.params['imagesize'], self.params['imagesize']) )
            input_fixed_arr[i] = input_fixed

        return input_fixed_arr

# params = {}
# params.update(defaultParams)
# print (params)
# emulate = omniglot_hd_emulation(params)
#
# inputs, labels, testlabel = emulate.read_inputs(emulate.read_input_dataset())
# input_fixed_arr = emulate.inputs_to_fixed(inputs)
#
# print (input_fixed_arr.shape)
#
# print (inputs.shape, labels, testlabel)
# dict1, tmpw, tmpalpha, tmpeta = emulate.read_weights()
# print_keys = "".join(str(key) + " " for key in dict1)
# print (print_keys)
# emulate.Network(img)


def train(parameters):
    # Setup the parameter dictionary
    params = {}
    params.update(defaultParams)
    print (params)

    # Create an object for the omniglot file
    emulate = omniglot_hd_emulation(params)

    # Copy the dataset into a tensor converted to a numpy array
    input_dataset = emulate.read_input_dataset()

    # Load the weights and the parameters of the network now
    dict1, tmpw, tmpalpha, tmpeta = emulate.read_weights()

    #Iterate the images over the network now
    for num_test_sample in range(params['no_test_iters']):
        inputs, labels, testlabel = emulate.read_inputs(input_dataset)
        output_vector = emulate.Network(inputs[0], 3, 2, dict1)
        output_vector_fp = emulate.Network_fp(inputs[0], 3, 2, dict1)
        print (output_vector.shape, output_vector_fp.shape)

        
    print ("the images went through the network")
    # print ("***************************************\n", output_vector)
    # print("\n ***************************************", output_vector_fp)

    #np.savetxt("vector_fp32.txt", np.reshape(output_vector,())
    # print (input_fixed_arr.shape)
    # print (inputs.shape, labels, testlabel)

    diff_ratio = np.divide(np.absolute(output_vector_fp - output_vector), np.absolute(output_vector_fp))
    print (diff_ratio)
    print ("The mean error is", np.mean(np.absolute(diff_ratio)))

    num_bins = 64
    output_vector = np.reshape(output_vector,(64))
    output_vector = output_vector/(np.amax(np.absolute(output_vector)))
    n, bins, patches = plt.hist(output_vector, num_bins, facecolor='blue', alpha=0.5)
    plt.show()

    output_vector_fp = np.reshape(output_vector_fp,(64))
    output_vector_fp = output_vector_fp/(np.amax(np.absolute(output_vector_fp)))
    n,bis,patches = plt.hist(output_vector_fp,num_bins, facecolor='blue', alpha =0.5)
    plt.show()
    print_keys = "".join(str(key) + " " for key in dict1)
    print (print_keys)

    #emulate.Network(img)

train(defaultParams)
