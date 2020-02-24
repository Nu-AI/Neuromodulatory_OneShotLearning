import pdb
import os
import platform
import torch
import pickle
import numpy as np

ttype = torch.cuda.FloatTensor;
suffix="_Wactiv_tanh_alpha_free_flare_0_gamma_0.666_imgsize_31_ipd_0_lr_3e-05_nbclasses_5_nbf_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_steplr_1000000.0_rngseed_"+"1"+"_5000000"
with open('../results/results'+suffix+'.dat', 'rb') as fo:
    tmpw = torch.nn.Parameter(torch.from_numpy(pickle.load(fo)).type(ttype))
    tmpalpha = torch.nn.Parameter(torch.from_numpy(pickle.load(fo)).type(ttype))
    tmpeta = torch.nn.Parameter(torch.from_numpy(pickle.load(fo)).type(ttype))

print (tmpw.shape, tmpalpha.shape, tmpeta.shape)
print ("The final layer weighs are ", tmpw)

dict = {}
dict = torch.load('../torchmodels/torchmodel'+suffix + '.txt')
new_dict={}
for i in dict:
    new_dict[i] = dict[i].cpu().numpy()
    print (i)
y = tmpw.cpu().detach().numpy()

print (new_dict['cv1.weight'].shape, new_dict['cv1.bias'].shape)
#print (dict['w'])
#print  (dict)
