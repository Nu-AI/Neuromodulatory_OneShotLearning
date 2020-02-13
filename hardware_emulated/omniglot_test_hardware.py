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


defaultParams = {

    'nbclasses': 5,
    'nbshots': 1,  # Number of 'shots' in the few-shots learning
    'prestime': 1,
    'nbf' : 64    ,
    'prestimetest': 1,
    'interpresdelay': 0,
    'imagesize': 31,    # 28*28
    'nbiter': 10000000,
    'learningrate': 1e-5,
    'print_every': 10,
    'rngseed':0,
    'quantization': 0
}
NBTESTCLASSES = 100

def gen_inputs_labels_target(params, imagedata):
        inputT = np.zeros((params['nbsteps'], 1, 1, params['imagesize'], params['imagesize']))    #inputTensor, initially in numpy format... Note dimensions: number of steps x batchsize (always 1) x NbChannels (also 1) x h x w
        labelT = np.zeros((params['nbsteps'], 1, params['nbclasses']))      #labelTensor, initially in numpy format...

        patterns=[]
        if test:
            cats = np.random.permutation(np.arange(len(imagedata) - NBTESTCLASSES, len(imagedata)))[:params['nbclasses']]  # Which categories to use for this *testing* episode?
        else:
            cats = np.random.permutation(np.arange(len(imagedata) - NBTESTCLASSES))[:params['nbclasses']]  # Which categories to use for this *training* episode?

        cats = np.random.permutation(cats)
        #print(cats)


        rots = np.random.randint(4, size=len(imagedata))


        testcat = random.choice(cats) # select the class on which we'll test in this episode
        unpermcats = cats.copy()


        location = 0
        for nc in range(params['nbshots']):
            np.random.shuffle(cats)   # Presentations occur in random order
            for ii, catnum in enumerate(cats):
                #print(catnum)
                p = random.choice(imagedata[catnum])
                for nr in range(rots[catnum]):
                    p = np.rot90(p)
                p = skimage.transform.resize(p, (31, 31))
                for nn in range(params['prestime']):


                    inputT[location][0][0][:][:] = p[:][:]
                    labelT[location][0][np.where(unpermcats == catnum)] = 1
                    #if nn == 0:
                    #    print(labelT[location][0])
                    location += 1
                location += params['interpresdelay']

        # Inserting the test character
        p = random.choice(imagedata[testcat])
        for nr in range(rots[testcat]):
            p = np.rot90(p)
        p = skimage.transform.resize(p, (31, 31))
        for nn in range(params['prestimetest']):
            inputT[location][0][0][:][:] = p[:][:]
            location += 1

        # Generating the test label
        testlabel = np.zeros(params['nbclasses'])
        testlabel[np.where(unpermcats == testcat)] = 1


        assert(location == params['nbsteps'])

        inputT = torch.from_numpy(inputT).type(ttype)  # Convert from numpy to Tensor
        labelT = torch.from_numpy(labelT).type(ttype)
        targetL = torch.from_numpy(testlabel).type(ttype)

        return inputT, labelT, targetL
