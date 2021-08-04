```
Developed by Anurag Daram at UTSA
email:anurag.daram@my.utsa.edu
```


import sys
import time

import numpy as np
import torch

from OSL_network.input_generator import input_generator
from OSL_network.model import Network
from OSL_network.parameters import params
from OSL_network.weight_loader import WeightLoader

# Initializing the objects for generating datasets and loading weights
input_retriever = input_generator(params)
weight_loader = WeightLoader(params)

# The network model
OSL_net = Network()

# Retrieving the input dataset
input_dataset = input_retriever.dataset_reader()

# The loss function, optimizer and the learning rate scheduler
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(OSL_net.parameters(), lr=1.0 * params['lr'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=params['gamma'], step_size=params['step_lr'])

curr_time = time.time()
accum_loss = 0.0
net_loss = []
# Running the training set for the number of episodes to be trained for num_iters
for num_iter in range(params['num_iters']):
	# Randomly sample the inputs, labels and targets
	inputs, labels, target = input_retriever.gen_train_test_split(input_dataset, test=0)
	trace = OSL_net.initalize_trace()
	# Sending in the training steps of the samples and updating the trace
	for _ in range(params['nb_steps']):
		feature_activ, output_activ, trace = OSL_net.forward(inputs['step'], labels['step'], trace)
		
	# Updating the loss of the network with respect to the sent in test sample
	loss = criterion(output_activ[0], target)

	# Computing the loss of the network
	curr_loss = loss.data
	accum_loss += curr_loss
	if not params['test']:
		loss.backward()
		scheduler.step()
		optimizer.step()
	else:
		print(num_iter, "====")
		td = target.cpu().numpy()
		yd = output_activ.data.cpu().numpy()[0]
		print("y: ", yd[:10])
		print("target: ", td[:10])
		# print("target: ", target.unsqueeze(0)[0][:10])
		absdiff = np.abs(td - yd)
		print("Mean / median / max abs diff:", np.mean(absdiff), np.median(absdiff), np.max(absdiff))
		print("Correlation (full / sign): ", np.corrcoef(td, yd)[0][1], np.corrcoef(np.sign(td), np.sign(yd))[0][1])

		previoustime = curr_time
		nowtime = time.time()
		print("Time spent on last", params['test_every'], "iters: ", nowtime - previoustime)

		print("Loss on single withheld-data episode:", curr_loss)
		net_loss.append(curr_loss)
		print("Eta: ", OSL_net.eta.data.cpu().numpy())
		sys.stdout.flush()
