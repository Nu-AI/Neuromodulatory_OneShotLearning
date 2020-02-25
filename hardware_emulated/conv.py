import numpy as np
import os, sys
import decimal

import skimage.data
from skimage.transform import rescale, resize

import ctypes, ctypes.util
from ctypes import cdll

##################################################################
# The next section involves setting up the C++ wrappers for the  #
# given convolutional operations in fixed point.                 #
##################################################################
Fixedlib = ctypes.CDLL("FixedPoint.so")

Fixedlib.Float_to_Fixed.restype = ctypes.c_int32
Fixedlib.Float_to_Fixed.argtypes = (ctypes.c_float, ctypes.c_int, ctypes.c_int)

Fixedlib.Fixed_to_Float.restype = ctypes.c_float
Fixedlib.Fixed_to_Float.argtypes = (ctypes.c_float, ctypes.c_int)

Fixedlib.Fixed_to_Float2.restype = ctypes.c_float
Fixedlib.Fixed_to_Float2.argtypes = (ctypes.c_float, ctypes.c_int)

Fixedlib.Fixed_Mul.restype = ctypes.c_int32
Fixedlib.Fixed_Mul.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_int)

Fixedlib.Fixed_ACC.restype = ctypes.c_float
Fixedlib.Fixed_ACC.argtypes = (np.ctypeslib.ndpointer(dtype=np.float32), ctypes.c_int)

def Float_to_Fixed(number, integer, fraction):
    result = Fixedlib.Float_to_Fixed(number, integer, fraction)
    return result

def Fixed_to_Float(number, fraction):
    result = Fixedlib.Fixed_to_Float(number, fraction)
    return result

def Fixed_to_Float2(number, fraction):
    result = Fixedlib.Fixed_to_Float2(number, fraction)
    return result

def Fixed_Mul(input1, input2, integer, fraction):
    result = Fixedlib.Fixed_Mul(input1, input2, integer, fraction)
    return result

def Fixed_ACC(Product, shape):
    result = Fixedlib.Fixed_ACC(Product, shape)
    return result

#Sample images and filters for testing the layer
image = skimage.data.chelsea()
img  = skimage.color.rgb2gray(image)
l1_filter = np.zeros((2,3,3))
l1_filter_new = np.zeros((2,3,3))
img = skimage.transform.resize(img, (31,31))
img = img/np.linalg.norm(img)
norm_image = map(lambda row:row/np.linalg.norm(row), img)
#print (norm_image, img, "This is the norm image")
print(img.shape, " %%%%%%%%%")

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

####################################################################
# This class will take in the following arguments with the input   #
# image being the input_map, the kernel size, the stride and  the  #
# number of filters for that particular layer.                     #
####################################################################

class conv_3_3:

    def __init__(self, input_map, kernel_size, stride, nbfilters):
        self.nbfilters = nbfilters
        self.input_map = input_map
        self.kernel_size = kernel_size
        self.stride = stride
        self.filter_array = np.random.randn(nbfilters, 3, 3) / 9

    ###############################################################
    # Convolution operation handler. Regulates the number of      #
    # convolutions taking place and is responsible for generating #
    # the output feature maps                                     #
    ###############################################################

    def forward(self, filter):

        feature_map_size = int((self.input_map.shape[1] - self.kernel_size) / (self.stride)) + 1
        temp_feature_map = np.zeros((feature_map_size, feature_map_size, self.nbfilters))

        #Starting the convolutions
        count = 0
        for ftr in range(self.nbfilters):
            #print ("Entered thhe looop *******************", self.nbfilters, ftr)
            temp_filter = filter[ftr,:]
            #print (temp_filter.shape, self.input_map.shape[-1])
            if (len(self.input_map.shape)>2):
                #print ("Larger map shape", self.input_map.shape)
                #reshaped_filter = np.reshape(filter, (3,3,2))
                conv_map = self.conv_perform(self.input_map[:,:,0], temp_filter[0,:,:])
                for channel in range(1, self.input_map.shape[-1]):
                    count += channel
                    #print ("Entering the inner loop for accumulating results from feature maps", (channel+ftr),self.input_map[:,:,channel].shape)
                    conv_map  = conv_map + self.conv_perform(self.input_map[:,:,channel], temp_filter[channel,:,:])
            else:
                conv_map = self.conv_perform(self.input_map,temp_filter)
            temp_feature_map[:,:,ftr] = conv_map

        return temp_feature_map

    ###############################################################
    # This method performs the actual convolutions. The selected  #
    # input map and the filters are sent into this function as    #
    # arguments and the output is calculated and returned as the  #
    # individual feature maps.                                    #
    ###############################################################

    def conv_perform(self, input_map, filter):

        point_wise_mult = 0

        temp_counter_1  = 0
        temp_counter_2  = 0
        feature_map_size = int((input_map.shape[1] - self.kernel_size) / (self.stride)) + 1
        feature_map = np.zeros((feature_map_size, feature_map_size))

        #Setting up in fixed and floating point
        point_wise_mult_fixed = 0
        val_in_float = 0.0
        float_acc = 0.0
        feature_map_float = np.zeros((feature_map_size, feature_map_size))
        feature_map_fixed = np.zeros((feature_map_size, feature_map_size))
        temp_array  = np.zeros((feature_map_size, feature_map_size))
        a = np.zeros((self.kernel_size*self.kernel_size),dtype=np.float32)
        input_map_size_row = input_map.shape[0]
        input_map_size_col = input_map.shape[1]
        for t in range(0, input_map_size_col - self.kernel_size+1, self.stride):
            for k in range (0, input_map_size_row-self.kernel_size+1, self.stride):
                for i in range (self.kernel_size):
                    for j in range (self.kernel_size):
                        point_wise_mult += input_map[k+i][t+j] * filter[i][j]
                        point_wise_mult_fixed += Fixed_Mul(input_map[k+i][t+j],filter[i][j],6,10)
                        a[i*j + j] = point_wise_mult_fixed # Testing for fixed accum
                        val_in_float = Fixed_to_Float2(point_wise_mult_fixed, 10)
                #Need to fix the fixed accumulator
                #float_acc = Fixed_ACC(a,self.kernel_size*self.kernel_size)
                feature_map[temp_counter_1][temp_counter_2] = point_wise_mult
                feature_map_fixed[temp_counter_1][temp_counter_2] = point_wise_mult_fixed
                feature_map_float[temp_counter_1][temp_counter_2] = val_in_float
                temp_counter_1 += 1
                point_wise_mult = 0
                point_wise_mult_fixed = 0
            temp_counter_2 += 1
            temp_counter_1 = 0
        temp_array = feature_map - feature_map_float
        #print (temp_array, "the difference in precision values")
        return feature_map_float

conv1 = conv_3_3(img, 3,2,2)
feature_maps = conv1.forward(l1_filter)
#print (feature_maps[:,:,0], feature_maps.shape)
conv2  = conv_3_3(feature_maps, 3,2,2)
new_feature_maps= conv2.forward(l2_filter)
#print(new_feature_maps,new_feature_maps.shape)
conv3  = conv_3_3(new_feature_maps, 3,2,2)
updated_f_maps = conv3.forward(l2_filter)
#print (updated_f_maps,updated_f_maps.shape)
conv4 = conv_3_3(updated_f_maps,3,2,2)
updated_final_maps = conv4.forward(l2_filter)
#print(updated_final_maps,updated_final_maps.shape)
