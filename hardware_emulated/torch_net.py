import torch
import numpy as np


class Network(nn.Module):
    def __init__(self, params):
        super(Network, self).__init__()
        self.rule = params['rule']
        if params['flare'] == 1:
            self.cv1 = torch.nn.Conv2d(1, params['nbf'] //4 , 3, stride=2).cuda()
            self.cv2 = torch.nn.Conv2d(params['nbf'] //4 , params['nbf'] //4 , 3, stride=2).cuda()
            self.cv3 = torch.nn.Conv2d(params['nbf'] //4, params['nbf'] //2, 3, stride=2).cuda()
            self.cv4 = torch.nn.Conv2d(params['nbf'] //2,  params['nbf'], 3, stride=2).cuda()
        else:
            self.cv1 = torch.nn.Conv2d(1, params['nbf'] , 3, stride=2).cuda()
            self.cv2 = torch.nn.Conv2d(params['nbf'] , params['nbf'] , 3, stride=2).cuda()
            self.cv3 = torch.nn.Conv2d(params['nbf'] , params['nbf'] , 3, stride=2).cuda()
            self.cv4 = torch.nn.Conv2d(params['nbf'] ,  params['nbf'], 3, stride=2).cuda()



        self.w =  torch.nn.Parameter((.01 * torch.randn(params['nbf'], params['nbclasses'])).cuda(), requires_grad=True)
        #self.w =  torch.nn.Parameter((.01 * torch.rand(params['plastsize'], params['nbclasses'])).cuda(), requires_grad=True)
        if params['alpha'] == 'free':
            self.alpha =  torch.nn.Parameter((.01 * torch.rand(params['nbf'], params['nbclasses'])).cuda(), requires_grad=True) # Note: rand rather than randn (all positive)
        elif params['alpha'] == 'yoked':
            self.alpha =  torch.nn.Parameter((.01 * torch.ones(1)).cuda(), requires_grad=True)
        else :
            raise ValueError("Must select a value for alpha ('free' or 'yoked')")
        self.eta = torch.nn.Parameter((.01 * torch.ones(1)).cuda(), requires_grad=True)  # Everyone has the same eta
        self.params = params

    def forward(self, inputx, inputlabel, hebb):
        if self.params['activ'] == 'selu':
            activ = F.selu(self.cv1(inputx))
            activ = F.selu(self.cv2(activ))
            activ = F.selu(self.cv3(activ))
            activ = F.selu(self.cv4(activ))
        elif self.params['activ'] == 'relu':
            activ = F.relu(self.cv1(inputx))
            activ = F.relu(self.cv2(activ))
            activ = F.relu(self.cv3(activ))
            activ = F.relu(self.cv4(activ))
        elif self.params['activ'] == 'tanh':
            activ = F.tanh(self.cv1(inputx))
            activ = F.tanh(self.cv2(activ))
            activ = F.tanh(self.cv3(                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          activ))
            activ = F.tanh(self.cv4(activ))
        else:
            raise ValueError("Parameter 'activ' is incorrect (must be tanh, relu or selu)")
        #activ = F.tanh(self.conv2plast(activ.view(1, self.params['nbf'])))
        #activin = activ.view(-1, self.params['plastsize'])
        activin = activ.view(-1, self.params['nbf'])

        if self.params['alpha'] == 'free':
            activ = activin.mm( self.w + torch.mul(self.alpha, hebb)) + 1000.0 * inputlabel # The expectation is that a nonzero inputlabel will overwhelm the inputs and clamp the outputs
        elif self.params['alpha'] == 'yoked':
            activ = activin.mm( self.w + self.alpha * hebb) + 1000.0 * inputlabel # The expectation is that a nonzero inputlabel will clip this and affect the other inputs
        activout = F.softmax( activ )

        if self.rule == 'hebb':
            hebb = (1 - self.eta) * hebb + self.eta * torch.bmm(activin.unsqueeze(2), activout.unsqueeze(1))[0] # bmm used to implement outer product; remember activs have a leading singleton dimension
        elif self.rule == 'oja':
            hebb = hebb + self.eta * torch.mul((activin[0].unsqueeze(1) - torch.mul(hebb , activout[0].unsqueeze(0))) , activout[0].unsqueeze(0))  # Oja's rule. Remember that yin, yout are row vectors (dim (1,N)). Also, broadcasting!
        else:
            raise ValueError("Must select one learning rule ('hebb' or 'oja')")

        return activout, hebb

    def initialZeroHebb(self):
        #return Variable(torch.zeros(self.params['plastsize'], self.params['nbclasses']).type(ttype))
        return Variable(torch.zeros(self.params['nbf'], self.params['nbclasses']).type(ttype))
