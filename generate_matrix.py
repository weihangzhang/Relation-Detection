from scipy import misc
import numpy as np
import tensorflow as tf
from load_voc import get_image_names
from pprint import pprint
import os
import random

import sys
sys.path.insert(0, os.path.abspath('./pyAIUtils-master/'))
from aiutils.tftools import layers
from aiutils.tftools import images
from aiutils.data import batch_creators
from aiutils.tftools import placeholder_management
from aiutils.tftools import batch_normalizer
from aiutils.tftools import var_collect
from aiutils.vis import image

class data_matrix:
	def __init__(self):
		self.image_matrix, self.label_matrix, self.label_list = self.get_matrix()

	def get_matrix(self):
		image_names = os.listdir('./voc_objects/')
		label_list = []

		for name in image_names:
			if '_' in name:
				continue
			name_s = name.split('-')
			if name_s[1] not in label_list:
				label_list.append(name_s[1])
		
		# define image matrix and label matrix
		image_matrix = np.zeros([len(image_names), 224, 224, 3])
		label_matrix = np.zeros([len(image_names), 11])
		
		for i in range(len(image_names)):
			print i
			name = image_names[i]
			if '_' in name:
				continue
			name_s = name.split('-')

			# read image, resize and assign to numpy matrix
			img = misc.imread('./voc_objects/' + name)
			img = image.imresize(img, method='bilinear', output_size=(224,224))
			image_matrix[i] = img
			# assign labeling
			label_matrix[i, label_list.index(name_s[1])] = 1

		# print image_matrix
		return image_matrix, label_matrix, label_list

	def get_next_batch(self, num):
		arr = range(len(self.label_matrix))
		random.shuffle(arr)
		chosen = arr[0:num]
		# print chosen
		# print len(image_matrix[chosen])
		return self.image_matrix[chosen], self.label_matrix[chosen]

# data = data_matrix()
# x, y = data.get_next_batch(64)
# print x.shape
# print y.shape