import pdb
import torch
import torch.nn as nn
from torch.autograd import Variable
import click
import numpy as np
from numpy import random
import torch.nn.functional as F
from torch import optim


class Network(nn.Module):
    def __init__(self, params):
        super(Network, self).__init__()
        # self.rule = params['rule']
        # if params['flare'] == 1:
        #     self.cv1 = torch.nn.Conv2d(1, params['no_filters'] //4 , 3, stride=2).cuda()
        #     self.cv2 = torch.nn.Conv2d(params['no_filters'] //4 , params['no_filters'] //4 , 3, stride=2).cuda()
        #     self.cv3 = torch.nn.Conv2d(params['no_filters'] //4, params['no_filters'] //2, 3, stride=2).cuda()
        #     self.cv4 = torch.nn.Conv2d(params['no_filters'] //2,  params['no_filters'], 3, stride=2).cuda()
        # else:
        self.cv1 = torch.nn.Conv2d(1, params['no_filters'] , 3, stride=2).cuda()
        self.cv2 = torch.nn.Conv2d(params['no_filters'] , params['no_filters'] , 3, stride=2).cuda()
        self.cv3 = torch.nn.Conv2d(params['no_filters'] , params['no_filters'] , 3, stride=2).cuda()
        self.cv4 = torch.nn.Conv2d(params['no_filters'] ,  params['no_filters'], 3, stride=2).cuda()
        self.w =  torch.nn.Parameter((.01 * torch.randn(params['no_filters'], params['no_classes'])).cuda(), requires_grad=True)
        #self.w =  torch.nn.Parameter((.01 * torch.rand(params['plastsize'], params['no_classes'])).cuda(), requires_grad=True)
        #if params['alpha'] == 'free':
        self.alpha =  torch.nn.Parameter((.01 * torch.rand(params['no_filters'], params['no_classes'])).cuda(), requires_grad=True) # Note: rand rather than randn (all positive)
        # elif params['alpha'] == 'yoked':
            # self.alpha =  torch.nn.Parameter((.01 * torch.ones(1)).cuda(), requires_grad=True)
        # else :
        # raise ValueError("Must select a value for alpha ('free' or 'yoked')")
        self.eta = torch.nn.Parameter((.01 * torch.ones(1)).cuda(), requires_grad=True)  # Everyone has the same eta
        self.params = params

    def forward(self, inputx, inputlabel, hebb):
        if self.params['activation'] == 'selu':
            activ = F.selu(self.cv1(inputx))
            activ = F.selu(self.cv2(activ))
            activ = F.selu(self.cv3(activ))
            activ = F.selu(self.cv4(activ))
        elif self.params['activation'] == 'relu':
            activ = F.relu(self.cv1(inputx))
            activ = F.relu(self.cv2(activ))
            activ = F.relu(self.cv3(activ))
            activ = F.relu(self.cv4(activ))
        elif self.params['activation'] == 'tanh':
            activ = F.tanh(self.cv1(inputx))
            activ = F.tanh(self.cv2(activ))
            activ = F.tanh(self.cv3(activ))
            activ = F.tanh(self.cv4(activ))
        else:
            raise ValueError("Parameter 'activation' is incorrect (must be tanh, relu or selu)")
        #activ = F.tanh(self.conv2plast(activ.view(1, self.params['no_filters'])))
        #activin = activ.view(-1, self.params['plastsize'])
        activin = activ.view(-1, self.params['no_filters'])

        # if self.params['alpha'] == 'free':
        activ = activin.mm( self.w + torch.mul(self.alpha, hebb)) + 1000.0 * inputlabel # The expectation is that a nonzero inputlabel will overwhelm the inputs and clamp the outputs
        # elif self.params['alpha'] == 'yoked':
            # activ = activin.mm( self.w + self.alpha * hebb) + 1000.0 * inputlabel # The expectation is that a nonzero inputlabel will clip this and affect the other inputs
        print ("\n",activ, "These are the intermediate activations before softmax function in the torchmodel \n %%%%%%%%%%%%%%%%%%")
        activout = F.softmax( activ )
        print ("\n",activout, "These are the intermediate activations after softmax function in the torchmodel \n %%%%%%%%%%%%%%%%%%")
        # if self.rule == 'hebb':
        #     hebb = (1 - self.eta) * hebb + self.eta * torch.bmm(activin.unsqueeze(2), activout.unsqueeze(1))[0] # bmm used to implement outer product; remember activs have a leading singleton dimension
        # elif self.rule == 'oja':
        #print ("\n", hebb, " \n The trace value before \n")
        hebb = hebb + self.eta * torch.mul((activin[0].unsqueeze(1) - torch.mul(hebb , activout[0].unsqueeze(0))) , activout[0].unsqueeze(0))  # Oja's rule. Remember that yin, yout are row vectors (dim (1,N)). Also, broadcasting!
        #print ("\n", hebb, " \n The trace value after \n")
        # else:
        #     raise ValueError("Must select one learning rule ('hebb' or 'oja')")
        print (self.eta, "The eta value being read from the results file")
        return activin, activout, hebb

    def initialZeroHebb(self):
        #return Variable(torch.zeros(self.params['plastsize'], self.params['no_classes']).type(ttype))
        return Variable(torch.zeros(self.params['no_filters'], self.params['no_classes']).type(torch.cuda.FloatTensor))
