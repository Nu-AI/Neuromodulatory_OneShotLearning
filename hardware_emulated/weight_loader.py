import pdb
import os
import platform
import torch
import pickle
import numpy as np

ttype = torch.cuda.FloatTensor;

#######################################################################################
# This method reads the final layer weights from a pickle file and converts the torch #
# tensor to nmupy arrays. Returns th weights, alpha param and learning factor         #
#######################################################################################

def read_fc_params(seed):

    suffix="_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_"+str(seed)+"_5000000"
    with open('../results/results'+suffix+'.dat', 'rb') as fo:
        tmpw = torch.nn.Parameter(torch.from_numpy(pickle.load(fo)).type(ttype))
        tmpalpha = torch.nn.Parameter(torch.from_numpy(pickle.load(fo)).type(ttype))
        tmpeta = torch.nn.Parameter(torch.from_numpy(pickle.load(fo)).type(ttype))
    dict = torch.load('../torchmodels/torchmodel'+suffix + '.txt')
    for i in dict:
        dict[i] = dict[i].cpu().numpy()
    return dict, tmpw.cpu().detach().numpy(), tmpalpha.cpu().detach().numpy(), tmpeta.cpu().detach().numpy()

dict1 = {}
dict1, tmpw, tmpalpha, tmpeta = read_fc_params(1)

print (dict1['cv1.weight'].shape, dict1['cv1.bias'].shape)
