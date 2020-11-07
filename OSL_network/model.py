import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):

	def __init__(self, params):
		super(Network, self).__init__()
		# Initializing the convolutional layers in the network
		self.cv1 = torch.nn.Conv2d(1, params['no_filters'], 3, stride=2).cuda()
		self.cv2 = torch.nn.Conv2d(params['no_filters'], params['no_filters'], 3, stride=2)
		self.cv3 = torch.nn.Conv2d(params['no_filters'], params['no_filters'], 3, stride=2)
		self.cv4 = torch.nn.Conv2d(params['no_filters'], params['no_filters'], 3, stride=2)

		# The weights parameter in the final layer
		self.w = torch.nn.Parameter((.01 * torch.randn(params['no_filters'], params['no_classes'])), requires_grad=True)

		# The modulatory learning parameter that influences the effect of the trace in the network
		self.alpha = torch.nn.Parameter((.01 * torch.rand(params['no_filters'], params['no_classes'])),
										requires_grad=True)
		# The selective learning rates for the network
		self.eta = torch.nn.Parameter((.01 * torch.ones(1)).cuda(), requires_grad=True)

		self.params = params

	def forward(self, input, inputlabel, trace):
		'''
		The forward pass of the network for the generation of
		:param input: The input sample to the model for forward pass
		:param inputlabel: The input label of the corresponding input sample
		:param trace: The trace value which will be updated
		:return feature_out: The output feature vector of the embedding
		:return activout: The output activations of the network
		:return trace: The trace value for updating the value in the network
		'''
		# Select the correct activation based on the user params
		if self.params['activation'] == 'relu':
			activ_func = lambda x: F.relu(x)
		elif self.params['activation'] == 'selu':
			activ_func = lambda x: F.selu(x)
		elif self.params['activation'] == 'tanh':
			activ_func = lambda x: F.tanh(x)
		else:
			raise ValueError("Enter correct activation ( relu, tanh and selu accepted )")

		# Forward pass across the feature embedding
		x = activ_func(self.cv1(input))
		x = activ_func(self.cv2(x))
		x = activ_func(self.cv3(x))
		x = activ_func(self.cv4(x))

		# Computing the output activations for the final layer
		feature_activ = x.view(-1, self.params['no_filters'])
		activout = feature_activ.mm(self.w + torch.mul(self.alpha, trace)) + 1000.0 * inputlabel

		# Updating the trace during the forward pass
		# The Oja base update of the network
		trace = trace + self.eta * torch.mul(
			(feature_activ[0].unsqueeze(1) - torch.mul(trace, activout[0].unsqueeze(0))),
			activout[0].unsqueeze(
				0))
		#TODO - Will have to add other trace update rules and evaluate their performance
		return feature_activ, activout, trace

	def initalize_trace(self):
		'''
		Initalizing the trace to zero before every episode in the training set.
		:return: The reinitialized trace of zero vectors
		'''
		return torch.zeros(self.params['no_filters'], self.params['no_classes'])

