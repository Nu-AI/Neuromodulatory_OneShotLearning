import numpy as np
import pdb
import pickle

from numpy import random
import random
import skimage
from skimage import transform
import os
import platform
import matplotlib.pyplot as plt

import click
import glob

defaultParams = {

	'no_classes': 5,  # Number of classes in the N-way K-shot learning case
	'no_shots': 1,  # Number of 'shots' in the few-shots learning
	'rand_seed': 0,  # Select the random seed file for taking the weights
	'no_filters': 64,  # Numebr of filters in the convolutional layers
	'imagesize': 31,  # The size of the 2D images to be reshaped to
	'present_test': 1,
	'learningrate': 1e-5,  # The initial learning rate for the network
	# 'print_every': 10,  # After how many epochs
}
TEST_CLASSES = 100


class input_generator:

	def __init__(self, params):
		self.params = params

	def dataset_reader(self, data_dir):
		train_dir = data_dir.join('images_background/')
		test_dir = data_dir.join('images_evaluation/')
		for curr_dir in (train_dir,test_dir):
			classdirs = glob.glob(curr_dir + '*')

class input_generator:

	def __init__(self, *args):
		print("Initialized the input_generator")
		for arg in args:
			print(arg)

	def dataset_reader(self, data_dir):
		train_dir = data_dir + 'images_background/'
		test_dir = data_dir + 'images_evaluation/'
		print(train_dir, test_dir)
		imagedata = []
		imagefilenames = []
		for curr_dir in (train_dir,
		                 test_dir):
			classdirs = glob.glob(curr_dir + '*')
			# print(classdirs[:4],"meoww")
			for class_dir in classdirs:
				imagedirs = glob.glob(class_dir + "/*")
				# print(chardirs)
				for image_dir in imagedirs:
					imgdata = []
					imgfiles = glob.glob(image_dir + '/*')
					# print (charfiles,"These are the charfiles")
					for file in imgfiles:
						# print(file,"the file data")
						filedata = plt.imread(file)
						# print(len(filedata))
						imgdata.append(filedata)
					imagedata.append(imgdata)
					imagefilenames.append(file)

		# imagedata[CharactertNumber][FileNumber] -> numpy(105,105)
		np.random.shuffle(imagedata)
		new_image_data = np.array(imagedata)
		print(len(imagedata), new_image_data.shape, 'this is the imagedata')
		print(imagedata[1][2].shape)
		print("Data loaded!")
		return imagedata

	def gen_inputs_labels_testlabel(self, params, imagedata, test):

		train_pick = np.arange(len(imagedata) - TEST_CLASSES, len(imagedata))
		test_pick = np.arange(len(imagedata) - TEST_CLASSES)

		if test:
			pick_samples = np.random.permutation(train_pick)[
			               :params['no_classes']]  # Which categories to use for this *testing* episode?
		else:
			pick_samples = np.random.permutation(test_pick)[
			               :params['no_classes']]  # Which categories to use for this *training* episode?

		pick_samples = np.random.permutation(pick_samples)  # Again randomizing

		inputs = np.zeros((params['steps'], params['imagesize'], params[
			'imagesize']))  # inputTensor, initially in numpy format... Note dimensions: number of steps x batchsize (always 1) x NbChannels (also 1) x h x w
		labels = np.zeros((params['steps'], params['no_classes']))  # labelTensor, initially in numpy format...
		testlabel = np.zeros(params['no_classes'])

		rotations = np.random.randint(4, size=len(imagedata))

		# select the class on which we'll test in this episode
		unpermuted_samples = pick_samples.copy()

		selection = 0
		for _ in range(params['no_shots']):

			np.random.shuffle(pick_samples)  # Always show the classes in fully random fashion
			for i, sample_num in enumerate(pick_samples):
				# Randomly select a sample
				p = random.choice(imagedata[sample_num])
				# Randomly rotate the seleted sample
				for _ in range(rotations[sample_num]):
					p = np.rot90(p)
				p = skimage.transform.resize(p, (31, 31))
				inputs[selection, :, :] = p[:][:]
				labels[selection][np.where(unpermuted_samples == sample_num)] = 1
				# if nn == 0:
				#    print(labelT[location][0])
				selection += 1

		# Inserting the test character
		test_sample = random.choice(unpermuted_samples)
		p = random.choice(imagedata[test_sample])
		for _ in range(rotations[test_sample]):
			p = np.rot90(p)
		p = skimage.transform.resize(p, (31, 31))

		# inputs[selection][0][0][:][:] = p[:][:]
		inputs[selection, :, :] = p[:][:]
		selection += 1

		# inputs = torch.from_numpy(inputs).type(torch.cuda.FloatTensor)  # Convert from numpy to Tensor
		# labels = torch.from_numpy(labels).type(torch.cuda.FloatTensor)
		# Generating the test label
		testlabel[np.where(unpermuted_samples == test_sample)] = 1

		assert (selection == params['steps'])

		return inputs, labels, testlabel


input = input_generator()
