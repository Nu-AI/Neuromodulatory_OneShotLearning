import torch
import pickle

class WeightLoader():

	def __init__(self, params):
		self.params = params

	def read_fc_params(self, seed):
		'''
		Read the weights from the pretrained model with the inputs to the network
		:param seed: The input random seed of the model run
		:return:
		'''
		if self.params['cuda']:
			ttype = torch.cuda.FloatTensor
		else:
			ttype = torch.FloatTensor
		suffix = "_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_" + str(
			seed) + "_5000000"
		with open('../results/results' + suffix + '.dat', 'rb') as fo:
			tmpw = torch.nn.Parameter(torch.from_numpy(pickle.load(fo)).type(ttype))
			tmpalpha = torch.nn.Parameter(torch.from_numpy(pickle.load(fo)).type(ttype))
			tmpeta = torch.nn.Parameter(torch.from_numpy(pickle.load(fo)).type(ttype))
		dict = torch.load('../torchmodels/torchmodel' + suffix + '.txt')
		for i in dict:
			dict[i] = dict[i].cpu().numpy()
		return dict, tmpw.cpu().detach().numpy(), tmpalpha.cpu().detach().numpy(), tmpeta.cpu().detach().numpy()
