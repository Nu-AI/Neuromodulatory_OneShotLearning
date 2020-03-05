import numpy as np
import os, sys
import decimal

import skimage.data
from skimage.transform import rescale, resize

import ctypes, ctypes.util
from ctypes import cdll
from conv import *

class FC_layer:

    def __init__(self, input_activs, inputlabel):
        self.input_activs = input_activs
        self.inputlabel = inputlabel

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

        for i in range(weights.shape[1]):
            for j in range(weights.shape[0]):
                activation[i] += self.input_activs[j]*plastic_wt[j][i]
                activation_fixed[i] += Fixed_mul(self.input_activs[j],plastic_wt_fixed[j][i],6,10)

            activation_fixed[i] = Fixed_to_Float2(activation_fixed[i],10)
