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
from OSL.hardware_emulated.conv import *
from OSL.hardware_emulated.weight_loader import *
from OSL.hardware_emulated.input_generator import *
from OSL.hardware_emulated.torch_model import *

from OSL.hardware_emulated.conv_fp32 import *
from OSL.hardware_emulated.fc_layer import *
import ctypes, ctypes.util
from ctypes import cdll

sns.set()
np.set_printoptions(threshold=np.inf)
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
# image = skimage.data.chelsea()
# img  = skimage.color.rgb2gray(image)
# l1_filter = np.zeros((2,3,3))
# l1_filter_new = np.zeros((2,3,3))
# img = skimage.transform.resize(img, (31,31))
# img = img/np.linalg.norm(img)
# norm_image = map(lambda row:row/np.linalg.norm(row), img)
#
#
# l1_filter[0, :, :] = np.array([[[-1, 0, 1],
#                                   [1, 0, 1],
#                                   [-1, 0, 1]]])
#
# l1_filter[1, :, :] = np.array([[[1,   1,  -1],
#                                 [0,   0,  0],
#                                [-1, 1, 1]]])
#
# l1_filter_new[0, :, :] = np.array([[[1, -1, 1],
#                                   [0, 0, 1],
#                                   [1, 0, -1]]])
#
# l1_filter_new[1, :, :] = np.array([[[1,   0,  -1],
#                                 [0,   1,  0],
#                                [-1, 0, 1]]])
#
# l2_filter =  np.zeros((2,2,3,3))
# l2_filter[0,:,:,:] = l1_filter
# l2_filter[1,:,:,:] = l1_filter_new


defaultParams = {
    'no_classes': 5,                                   # Number of classes in the N-way K-shot learning case
    'no_shots': 1,                                     # Number of 'shots' in the few-shot learning
    'rand_seed':0,                                     # Select the random seed file for taking the weights
    'no_filters' : 64,                                 # Number of filters in the convolutional layers
    'imagesize': 31,                                   # The size of the 2D images to be reshaped to
    'learningrate': 1e-5,                              # The initial learning rate for the network
    'present_test': 1,                                 # Number of times we present the testing class
    'no_test_iters': 1,
    'activation': 'tanh',
    'precision': 16,
    'fractional':10,
    'address' : '../omniglot_dataset/omniglot/python/', # enter the path of the dataset here
    #'print_every': 10,  # After how many epochs
}
TEST_CLASSES = 100


class omniglot_hd_emulation:


    def __init__(self, params):
        self.params = params
        self.params['steps'] =  params['no_shots']*(params['no_classes']) + params['present_test']
        self.params['decimal'] = params['precision'] - params['fractional']
        print ("Started the emulation")

    def read_input_dataset(self):
        input_val = input_generator()
        inputdata = input_val.dataset_reader(self.params['address'])
        print (np.array(inputdata).shape,"the shape of the input data")
        return inputdata

    def read_inputs(self, dataset):
        input_val = input_generator()
        return input_val.gen_inputs_labels_testlabel(self.params, dataset, test=False)

    def read_weights(self):
        load_weights = weight_loader()
        dict1, tmpw, tmpalpha, tmpeta =weight_load.read_fc_params(1)
        print ("Done loading weights")
        return dict1, tmpw, tmpalpha, tmpeta

    def init_net_torch(self):
        suffix="_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_"+str(1)+"_5000000"
        net = Network(self.params)
        net.load_state_dict(torch.load('../torchmodels/torchmodel'+ suffix + '.txt'))
        return net

    def init_mod_param(self,net):
        return net.initialZeroHebb()

    def Network(self, img, kernel_size, stride,dict):
        conv1 = conv_3_3(img, kernel_size,stride,self.params['no_filters'],dict['cv1.bias'],self.params['activation'],self.params['fractional'], self.params['decimal'])
        conv2 = conv_3_3(conv1.forward(dict['cv1.weight']), kernel_size,stride,self.params['no_filters'], dict['cv2.bias'],self.params['activation'],self.params['fractional'], self.params['decimal'])
        conv3 = conv_3_3(conv2.forward(dict['cv2.weight']), kernel_size,stride,self.params['no_filters'], dict['cv3.bias'],self.params['activation'],self.params['fractional'],self.params['decimal'])
        conv4 = conv_3_3(conv3.forward(dict['cv3.weight']), kernel_size,stride,self.params['no_filters'], dict['cv4.bias'],self.params['activation'],self.params['fractional'],self.params['decimal'])
        #print (conv4.forward(dict['cv4.weight']), "This is the final conv layer output"
        return conv4.forward(dict['cv4.weight'])

    def Network_fp(self, img, kernel_size, stride, dict):
        conv1 = conv_3_3_fp(img, kernel_size,stride,self.params['no_filters'],dict['cv1.bias'], self.params['activation'])
        conv2 = conv_3_3_fp(conv1.forward(dict['cv1.weight']), kernel_size,stride,self.params['no_filters'], dict['cv2.bias'],self.params['activation'])
        conv3 = conv_3_3_fp(conv2.forward(dict['cv2.weight']), kernel_size,stride,self.params['no_filters'], dict['cv3.bias'],self.params['activation'])
        conv4 = conv_3_3_fp(conv3.forward(dict['cv3.weight']), kernel_size,stride,self.params['no_filters'], dict['cv4.bias'], self.params['activation'])
        #print (conv4.forward(dict['cv4.weight']), "This is the final conv layer output")
        return conv4.forward(dict['cv4.weight'])

    def plastic_layer(self,input_activations,label,dict,mod):
        fully_connected = FC_layer(input_activations, label, dict['eta'], self.params['fractional'], self.params['decimal'])
        #mod = np.zeros_like(dict['w'])
        output =fully_connected.softmax(fully_connected.forward(dict['w'],dict['alpha'],mod))
        mod = fully_connected.update_trace(output,mod)
        return output,mod

    def torch_plastic_output(self, input_activations, label, mod_torch):
        input_activations = np.reshape(input_activations, (1,1,self.params['imagesize'], self.params['imagesize']))
        input_activations = torch.from_numpy(input_activations).type(torch.cuda.FloatTensor)
        label = np.reshape(label, (1,self.params['no_classes']))
        label = torch.from_numpy(label).type(torch.cuda.FloatTensor)
        #net = Network(self.params)
        #output_vector, final_out, mod_torch = net(Variable(input_activations, requires_grad=False), Variable(label, requires_grad=False), mod_torch)
        return input_activations, label

    def inputs_to_fixed(self, inputs):
        input_fixed_arr = np.empty_like(inputs)
        for i in range(self.params['steps']):
            input_list = list(np.reshape(inputs[i,:,:],(self.params['imagesize']*self.params['imagesize'])))
            input_fixed = list(map(lambda x: Float_to_Fixed(x,2,12), input_list))
            input_fixed = np.reshape(np.array(input_fixed),(self.params['imagesize'], self.params['imagesize']) )
            input_fixed_arr[i] = input_fixed
        return input_fixed_arr


def train(parameters):
    # Setup the parameter dictionary
    params = {}
    new_params = {}
    new_params.update(defaultParams)
    params.update(defaultParams)
    print (params)


    suffix="_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_"+str(1)+"_5000000"
    with open('../results/results'+suffix+'.dat', 'rb') as fo:
        tmpw = torch.nn.Parameter(torch.from_numpy(pickle.load(fo)).type(torch.cuda.FloatTensor))
        tmpalpha = torch.nn.Parameter(torch.from_numpy(pickle.load(fo)).type(torch.cuda.FloatTensor))
        tmpeta = torch.nn.Parameter(torch.from_numpy(pickle.load(fo)).type(torch.cuda.FloatTensor))

        tmplss = pickle.load(fo)
        paramdictLoadedFromFile = pickle.load(fo)
    new_params.update(paramdictLoadedFromFile)
    # Create an object for the omniglot file
    emulate = omniglot_hd_emulation(params)

    # Copy the dataset into a tensor converted to a numpy array
    input_dataset = emulate.read_input_dataset()

    # Load the weights and the parameters of the network now
    dict1, tmpw, tmpalpha, tmpeta = emulate.read_weights()

    print (dict1['alpha'].shape,dict1['w'].shape)
    temp_arr = np.zeros_like(dict1['w'])
    print(temp_arr.shape)
    acc_count = 0
    new_acc_count = 0
    net = Network(new_params)
    net.load_state_dict(torch.load('../torchmodels/torchmodel'+suffix + '.txt'))
    #net = emulate.init_net_torch()
    #Iterate the images over the network now
    for num_test_sample in range(params['no_test_iters']):
        mod = np.zeros_like(dict1['w'])
        #mod_torch = emulate.init_mod_param(net)
        mod_torch = net.initialZeroHebb()
        inputs, labels, testlabel = emulate.read_inputs(input_dataset)
        #print (inputs.shape, "This is the shape of the inputs\n", inputs.max(axis=0), inputs.max(axis=1), inputs.min(axis=0), inputs.min(axis=1))
        final_out = np.zeros_like(testlabel)
        for i in range(inputs.shape[0]):
            output_vector = emulate.Network(inputs[i], 3, 2, dict1)
            output_vector_fp = emulate.Network_fp(inputs[i], 3, 2, dict1)
            output_vector_fp = np.reshape(output_vector_fp,(params['no_filters']))

            output_vector = np.reshape(output_vector,(params['no_filters']))
            final_out,mod = emulate.plastic_layer(output_vector, labels[i], dict1, mod)
            print (dict1['eta'], "The eta value used in the emulated network \n")

            input_activations, label = emulate.torch_plastic_output(inputs[i], labels[i], mod_torch)
            torch_output_vector, torch_final_out, mod_torch = net(Variable(input_activations, requires_grad=False), Variable(label, requires_grad=False), mod_torch)

            print (output_vector.shape, output_vector_fp.shape,final_out,labels[i], testlabel)
            print ("torch outputs")
            print (torch_output_vector.shape, torch_final_out,"\n#####################\n")

            new_output_vector = torch_output_vector.cpu().detach().numpy().reshape((64))
            #final_out,mod = emulate.plastic_layer(output_vector_fp, labels[i], dict1, mod)
            difference = new_output_vector - output_vector_fp
            # if (i==5):
            #     print ("the difference in activations \n \n", difference)

            new_mod_torch = mod_torch.cpu().detach().numpy()
            trace_diff = new_mod_torch - mod
            if (i==4):
                print ("The difference in traces \n",trace_diff)

            final_weights = dict1['w']
            final_alpha = dict1['alpha']
            if (i==5):
                print ("\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
        if (np.argmax(final_out) == np.argmax(testlabel)):
            acc_count += 1
            print ("=====>",acc_count)
        else:
            print ("Mistake")
        if (np.argmax(torch_final_out.data.cpu().numpy()[0]) == np.argmax(testlabel)):
            new_acc_count +=1
        else:
            print ("Original mistake")
    print ("the images went through the network")
    print ("=====>",acc_count, "=====>", new_acc_count)
    # print ("***************************************\n", output_vector)
    # print("\n ***************************************", output_vector_fp)
    #np.savetxt("vector_fp32.txt", np.reshape(output_vector,())

    diff_ratio = np.divide(np.absolute(output_vector_fp - output_vector), np.absolute(output_vector_fp))
    #print (diff_ratio)
    print ("The mean error is", np.mean(np.absolute(diff_ratio)))

    fig, ax = plt.subplots(1,2,tight_layout=True)

    num_bins = 64
    output_vector = np.reshape(output_vector,(64))
    output_vector = output_vector/(np.amax(np.absolute(output_vector)))
    n, bins, patches = ax[0].hist(output_vector, num_bins, facecolor='blue', alpha=0.5)

    output_vector_fp = np.reshape(output_vector_fp,(64))
    output_vector_fp = output_vector_fp/(np.amax(np.absolute(output_vector_fp)))
    n,bis,patches = ax[1].hist(output_vector_fp,num_bins, facecolor='green', alpha =0.5)
    ax[0].set_ylabel("Count", fontsize = 15)
    ax[1].set_xlabel("Normalized Weights (Full Precision)")
    ax[0].set_xlabel("Normalized Weights (Quantized)" )
    plt.show()
    print_keys = "".join(str(key) + " " for key in dict1)

    print (print_keys)


train(defaultParams)
