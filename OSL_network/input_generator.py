import numpy as np
import pdb
import pickle

from numpy import random
import random
import skimage
from skimage import transform
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
		'''
		Read the omniglot dataset and return a array of the full dataset
		:param data_dir: The path to the omniglot dataset folder
		:return inputs: The input image dataset for the network
		:return labels: The corresponding labels for the input images
		:return dataset_array: The array consisiting of the entire dataset:
		'''
		train_dir = data_dir.join('images_background/')
		test_dir = data_dir.join('images_evaluation/')
		dataset_array = []
		imagefilenames = []
		for curr_dir in (train_dir,test_dir):
			classdirs = glob.glob(curr_dir + '*')
			for class_dir in classdirs:
				imagedirs = glob.glob(class_dir + '/*')
				for image_dir in imagedirs:
					imagedata= []
					imagefiles = glob.glob(image_dir + '\*')
					for file in imagefiles:
						filedata = plt.imread(file)
						imagedata.append(filedata)
					dataset_array.append(imagedata)
		np.random.shuffle(dataset_array)
		return dataset_array

	def gen_test_split(self, dataset_array, test):
		train_pick = np.arange(len(dataset_array) - TEST_CLASSES, len(dataset_array))
		test_pick = np.arange(len(dataset_array) - TEST_CLASSES)

		if test:
			pick_samples = np.random.permutation(train_pick)[
			               :self.params['no_classes']]  # Which categories to use for this *testing* episode?
		else:
			pick_samples = np.random.permutation(test_pick)[
			               :self.params['no_classes']]  # Which categories to use for this *training* episode?

		pick_samples = np.random.permutation(pick_samples)  # Again randomizing

		inputs = np.zeros((self.params['steps'], self.params['imagesize'], self.params[
			'imagesize']))  # inputTensor, initially in numpy format... Note dimensions: number of steps x batchsize (always 1) x NbChannels (also 1) x h x w
		labels = np.zeros((self.params['steps'], self.params['no_classes']))  # labelTensor, initially in numpy format...
		testlabel = np.zeros(self.params['no_classes'])

		rotations = np.random.randint(4, size=len(dataset_array))

		# select the class on which we'll test in this episode
		unpermuted_samples = pick_samples.copy()

		selection = 0
		for _ in range(self.params['no_shots']):

			np.random.shuffle(pick_samples)  # Always show the classes in fully random fashion
			for i, sample_num in enumerate(pick_samples):
				# Randomly select a sample
				p = random.choice(dataset_array[sample_num])
				# Randomly rotate the seleted sample
				for _ in range(rotations[sample_num]):
					p = np.rot90(p)
				p = skimage.transform.resize(p, (31, 31))
				inputs[selection, :, :] = p[:][:]
				labels[selection][np.where(unpermuted_samples == sample_num)] = 1
				selection += 1

		# Inserting the test character
		test_sample = random.choice(unpermuted_samples)
		p = random.choice(dataset_array[test_sample])
		for _ in range(rotations[test_sample]):
			p = np.rot90(p)
		p = skimage.transform.resize(p, (31, 31))

		inputs[selection, :, :] = p[:][:]
		selection += 1

		# inputs = torch.from_numpy(inputs).type(torch.cuda.FloatTensor)  # Convert from numpy to Tensor
		# labels = torch.from_numpy(labels).type(torch.cuda.FloatTensor)
		# Generating the test label
		testlabel[np.where(unpermuted_samples == test_sample)] = 1
		assert (selection == self.params['steps'])

		return inputs, labels, testlabel


input = input_generator()
