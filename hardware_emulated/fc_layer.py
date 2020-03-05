import numpy as np
import os, sys
import decimal

import skimage.data
from skimage.transform import rescale, resize

import ctypes, ctypes.util
from ctypes import cdll
from conv import *

class FC_layer:

    def __init__(self, input_activs, inputlabel, eta):
        self.input_activs = input_activs
        self.inputlabel = inputlabel
        self.eta = eta

    def forward(self, weights, alpha, mod):
        mod = np.zeros_like(weights)
        plastic_wt = np.zeros_like(weights)
        plastic_wt_fixed = np.zeros_like(weights)
        activation = np.zeros((weights.shape[1]))
        for i in  range(weights.shape[1]):
            for j in range(weights.shape[0]):
                plastic_wt[j][i] = weights[j][i] + alpha[j][i]*mod[j][i]
                plastic_wt_fixed[j][i] = Float_to_Fixed(weights[j][i],6,10) + Fixed_Mul(alpha[j][i],mod[j][i],6,10)
                plastic_wt_fixed[j][i] = Fixed_to_Float2(plastic_wt_fixed[j][i],10)
                # Alternate method
                #plastic_wt_fixed[j][i] = weights[j][i] + Fixed_to_Float2(Fixed_Mul(alpha[j][i],mod[j][i],6,10),10)
        for i in range(weights.shape[1]):
            for j in range(weights.shape[0]):
                activation[i] += self.input_activs[j]*plastic_wt[j][i]
                activation_fixed[i] += Fixed_mul(self.input_activs[j],plastic_wt_fixed[j][i],6,10)

            activation_fixed[i] = Fixed_to_Float2(activation_fixed[i],10)
            if(inputlabel[i]==1):
                activation[i] = activation[i]+1000
                activation_fixed[i] = activation_fixed[i] + 1000
        return activation


    def softmax(self, activation):
        return np.array(list(map(lambda x: 0 if x<500 else 1, activation)))

    def update_trace(self, activation_out, mod):
        mod_fixed = np.zeros_like(mod)
        for i in range(mod.shape[0]):
            inter2 = Float_to_Fixed(self.input_activs[i])
            for j in range(mod.shape[1]):
                if (activation_out[j] != 0):
                    mod[i][j] = mod[i][j] + self.eta*(activation_out[j])*(self.input_activs[i] \
                    - activation_out[j]*mod[i][j])
                    temp = Fixed_Mul(activation_out[j],mod[i][j],6,10)
                    temp2 = Fixed_to_Float2((inter2 - temp),10)
                    temp3 = Fixed_to_Float2(Fixed_Mul(self.eta,activation_out[j],6,10),10)
                    mod_fixed[i][j] = Float_to_Fixed(mod[i][j]) + Fixed_mul(temp2, temp3, 6 ,10)
                    mod_fixed[i][j] = Fixed_to_Float2(mod_fixed[i][j])
# alternate solution
                     #mod_fixed[i][j] = mod[i][j] + Fixed_to_Float2(Fixed_mul(temp2, temp3, 6 ,10),10)
                else:
                     #mod[i][j] = mod[i][j]
                     mod_fixed[i][j] = mod[i][j]

        return mod
