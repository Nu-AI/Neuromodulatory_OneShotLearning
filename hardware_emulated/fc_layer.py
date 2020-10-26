import numpy as np
import os, sys
import decimal

import skimage.data
from skimage.transform import rescale, resize

import ctypes, ctypes.util
from ctypes import cdll
from conv import *

class FC_layer:

    def __init__(self, input_activs, inputlabel, eta, fractional, decimal):
        self.input_activs = input_activs
        self.inputlabel = inputlabel
        self.eta = eta
        self.fractional = fractional
        self.decimal = decimal

    def forward(self, weights, alpha, mod):
        #mod = np.zeros_like(weights)
        plastic_wt = np.zeros_like(weights)
        plastic_wt_fixed = np.zeros_like(weights)
        activation = np.zeros((weights.shape[1]))
        activation_fixed = np.zeros((weights.shape[1]))
        for i in  range(weights.shape[1]):
            for j in range(weights.shape[0]):
                plastic_wt[j][i] = weights[j][i] + alpha[j][i]*mod[j][i]
                plastic_wt_fixed[j][i] = Float_to_Fixed(weights[j][i],self.decimal,self.fractional) + Fixed_Mul(alpha[j][i],mod[j][i],self.decimal,self.fractional)
                plastic_wt_fixed[j][i] = Fixed_to_Float2(plastic_wt_fixed[j][i],self.fractional)
                # Alternate method
                #plastic_wt_fixed[j][i] = weights[j][i] + Fixed_to_Float2(Fixed_Mul(alpha[j][i],mod[j][i],self.decimal,self.fractional),self.fractional)

        for i in range(weights.shape[1]):
            for j in range(weights.shape[0]):
                activation[i] += self.input_activs[j]*plastic_wt[j][i]
                activation_fixed[i] += Fixed_Mul(self.input_activs[j],plastic_wt_fixed[j][i],self.decimal,self.fractional)

            activation_fixed[i] = Fixed_to_Float2(activation_fixed[i],self.fractional)
        print (activation, "These are the intermediate activations")
        for i in range(len(self.inputlabel)):
            if(self.inputlabel[i]==1):
                activation[i] = activation[i]+1000
                activation_fixed[i] = activation_fixed[i] + 1000

        return activation
        #return activation_fixed


    def softmax(self, activation):
        convert_arr =  np.array(list(map(lambda x: 0 if x<500 else 1, activation)))
        if (not np.any(convert_arr)):
            convert_arr[np.argmax(activation)] = 1
        return convert_arr

    def update_trace(self, activation_out, mod):
        #mod_fixed = np.zeros_like(mod)
        # for i in range(mod.shape[0]):
        #     inter2 = Float_to_Fixed(self.input_activs[i],self.decimal,10)
        #     for j in range(mod.shape[1]):
        #         if (activation_out[j] != 0):
        #             mod[i][j] = mod[i][j] + self.eta*(activation_out[j])*(self.input_activs[i] \
        #             - activation_out[j]*mod[i][j])
        #             temp = Fixed_Mul(activation_out[j],mod[i][j],self.decimal,10)
        #             temp2 = Fixed_to_Float2((inter2 - temp),10)
        #             temp3 = Fixed_to_Float2(Fixed_Mul(self.eta,activation_out[j],self.decimal,10),10)
        #             mod_fixed[i][j] = Float_to_Fixed(mod[i][j] + Fixed_Mul(temp2, temp3, self.decimal ,10),self.decimal,10)
        #             mod_fixed[i][j] = Fixed_to_Float2(mod_fixed[i][j],10)
        #             # alternate solution
        #             #mod_fixed[i][j] = mod[i][j] + Fixed_to_Float2(Fixed_mul(temp2, temp3, self.decimal ,10),10)
        #         else:
        #             #mod[i][j] = mod[i][j]
        #              mod_fixed[i][j] = mod[i][j]
        inter_mod = np.zeros((mod.shape[0],1))
        inter_mod_fixed = np.zeros((mod.shape[0],1))
        inter_temp = 0
        inter_temp_fixed=0
        #These loops are calculating the inner multiplicative factors
        # These take the hadamard product of the modulatry trace with the output input_activations
        # Then the resultant matrix is subtracted from the input activations to the layer (feature vector)
        for i in range(mod.shape[0]):
            for j in range(mod.shape[1]):
                inter_temp += mod[i][j]*activation_out[j]   # accumulating partial sum

                inter_temp_fixed = Float_to_Fixed(inter_temp_fixed,self.decimal,self.fractional) + Fixed_Mul(mod[i][j],activation_out[j],self.decimal,self.fractional)

            inter_mod[i] = self.input_activs[i] - inter_temp

            inter_mod_fixed[i] = Float_to_Fixed(self.input_activs[i],self.decimal,self.fractional) - inter_temp_fixed

            inter_temp = 0
            inter_temp_fixed = 0

        for i in range(mod.shape[0]):
            for j in range(mod.shape[1]):
                temp = inter_mod[i]*activation_out[j]   # saving the intermediate value
                temp_fixed = Fixed_Mul(inter_mod[i],activation_out[j], self.decimal, self.fractional)

                temp_fixed = Fixed_to_Float2(temp_fixed,self.fractional)
                mod[i][j] = mod[i][j] + self.eta*temp
                #mod_fixed = Float_to_Fixed(mod[i][j],self.decimal,self.fractional) + Fixed_Mul(self.eta,temp_fixed, self.decimal,self.fractional)
                # Incorporate fixed point logic in the code
                #mod[i][j] = Fixed_to_Float2(mod_fixed, self.fractional)
        #print ("\n\n",mod,"\nThis is the emulated trace value after learning \n\n")

        return mod
