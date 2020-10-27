import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Network(nn.Module):

	def __init__(self, params):
		super(Network, self).__init__()
		# Initializing the convolutional layers in the network
		self.cv1 = torch.nn.Conv2d(1, params['no_filters'], 3, stride=2).cuda()
		self.cv2 = torch.nn.Conv2d(params['no_filters'], params['no_filters'], 3, stride=2)
		self.cv3 = torch.nn.Conv2d(params['no_filters'], params['no_filters'], 3, stride=2)
		self.cv4 = torch.nn.Conv2d(params['no_filters'], params['no_filters'], 3, stride=2)
		# The weights parameter in the final layer
		self.w = torch.nn.Parameter((.01 * torch.randn(params['no_filters'], params['no_classes'])),
		                            requires_grad=True)
		# The modulatory learning parameter that influences the effect of the trace in the network
		self.alpha = torch.nn.Parameter((.01 * torch.rand(params['no_filters'], params['no_classes'])),
		                                requires_grad=True)
		# The selective learning rates for the network
		self.eta = torch.nn.Parameter((.01 * torch.ones(1)).cuda(), requires_grad=True)

		self.params = params

	def forward(self, input, inputlabel, trace):
		if self.params['activation'] == 'relu':
			activ_func = lambda x: F.relu(x)
		elif self.params['activation'] == 'selu':
			activ_func = lambda x: F.selu(x)
		elif self.params['activation'] == 'tanh':
			activ_func = lambda x: F.tanh(x)
		else:
			raise ValueError("Enter correct activation ( relu, tanh and selu accepted )")

		x = activ_func(self.cv1(input))
		x = activ_func(self.cv2(x))
		x = activ_func(self.cv3(x))
		x = activ_func(self.cv4(x))

		feature_activ = x.view(-1, self.params['no_filters'])
		activout = feature_activ.mm(self.w + torch.mul(self.alpha, trace)) + 1000.0 * inputlabel

		trace = trace + self.eta * torch.mul(
			(feature_activ[0].unsqueeze(1) - torch.mul(trace, activout[0].unsqueeze(0))),
			activout[0].unsqueeze(
				0))
		return feature_activ, activout, trace

	def initalize_trace(self):
		return Variable(torch.zeros(self.params['no_filters'], self.params['no_classes']))
