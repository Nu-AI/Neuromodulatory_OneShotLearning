import numpy as np
import os, sys
import decimal


import skimage.data
from skimage.transform import rescale, resize
image = skimage.data.chelsea()
img  = skimage.color.rgb2gray(image)
l1_filter = np.zeros((2,3,3))
img = skimage.transform.resize(img, (31,31))
print(img.shape, " %%%%%%%%%")

l1_filter[0, :, :] = np.array([[[-1, 0, 1],
                                  [-1, 0, 1],
                                  [-1, 0, 1]]])

l1_filter[1, :, :] = np.array([[[1,   1,  1],
                                [0,   0,  0],
                               [-1, -1, -1]]])
l2_filter =  np.zeros((2,2,3,3))
l2_filter[0,:,:,:] = l1_filter
l2_filter[1,:,:,:] = l1_filter

class conv_3_3:

    def __init__(self, input_map, kernel_size, stride, nbfilters):
        self.nbfilters = nbfilters
        self.input_map = input_map
        self.kernel_size = kernel_size
        self.stride = stride
        self.filter_array = np.random.randn(nbfilters, 3, 3) / 9


    def forward(self, filter):
        # if (len(filter.shape)>3):
        #     if (self.input_map.shape[-1] != filter.shape[-1]):
        #         print ("The shapes must match for the image and the filter")
        #         sys.exit()
        # if filter.shape[1] != filter.shape[2]:
        #     print ("The filter must have the same dimensions")
        #     sys.exit()

        feature_map_size = int((self.input_map.shape[1] - self.kernel_size) / (self.stride)) + 1
        temp_feature_map = np.zeros((feature_map_size, feature_map_size, self.nbfilters))

        #Starting the convolutions
        for ftr in range(self.nbfilters):
            print ("Entered thhe looop *******************", self.nbfilters, ftr)
            temp_filter = filter[ftr,:]
            print (temp_filter.shape)
            if (len(self.input_map.shape)>2):
                print ("Larger map shape", self.input_map.shape)
                #reshaped_filter = np.reshape(filter, (3,3,2))
                conv_map = self.conv_perform(self.input_map[:,:,0], temp_filter[0,:,:])
                for channel in range(1, self.input_map.shape[-1]):
                    conv_map  = conv_map + self.conv_perform(self.input_map[:,:,channel], temp_filter[channel,:,:])
            else:
                conv_map = self.conv_perform(self.input_map,temp_filter)
            temp_feature_map[:,:,ftr] = conv_map

        return temp_feature_map


    def conv_perform(self, input_map, filter):

        point_wise_mult = 0
        temp_counter_1  = 0
        temp_counter_2  = 0
        feature_map_size = int((input_map.shape[1] - self.kernel_size) / (self.stride)) + 1
        feature_map = np.zeros((feature_map_size, feature_map_size))
        #print (feature_map_size, " *******************")
        # if(len(input_map.shape)> 2 ):
        #     input_map_size_row = input_map.shape[0]
        #     input_map_size_col = input_map.shape[1]
        # else:
        input_map_size_row = input_map.shape[0]
        input_map_size_col = input_map.shape[1]
        for t in range(0,input_map_size_col - 1, self.stride):

            for k in range (0, input_map_size_row-1, self.stride):
                for i in range(self.kernel_size):
                    for j in range (self.kernel_size):
                        #Refactor this line to work for varied input shapes
                        point_wise_mult += input_map[i][j] * filter[i][j]

                feature_map[temp_counter_1][temp_counter_2] = point_wise_mult
                temp_counter_1 +=1
            temp_counter_2 += 1
            temp_counter_1 = 0
        return feature_map

c = conv_3_3(img, 3,2,2)
feature_maps = c.forward(l1_filter)
print (feature_maps.shape)#, feature_maps)
d  = conv_3_3(feature_maps, 3,2,2)
new_feature_maps= d.forward(l2_filter)
print(new_feature_maps.shape)
