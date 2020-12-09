import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

def convert_txt_to_array(path, arr):
	'''
	path - path of the text file
	arr - matrix with appropriate shape as that of the text file

	returns a numpy array with the given pixel values in the text file
	'''
	file = open(path, "r")
	z = 0
	for line in file:
		arr[z, :] = np.array(line.split(',')).astype('float')
		z += 1

	return arr

