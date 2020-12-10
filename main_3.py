import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import seaborn as sns
sns.set_palette('hls', 10)

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

path_ball = "/content/drive/MyDrive/Computational Neuroscience/ball.txt"
path_cat = "/content/drive/MyDrive/Computational Neuroscience/cat.txt"
path_mona = "/content/drive/MyDrive/Computational Neuroscience/mona.txt"

ball_arr = convert_txt_to_array(path_ball, np.zeros((90, 100)))
cat_arr = convert_txt_to_array(path_cat, np.zeros((90, 100)))
cat_arr = np.where(cat_arr < 0, -1, 1).astype('float')
mona_arr = convert_txt_to_array(path_mona, np.zeros((90, 100)))

N = ball_arr.shape[0]*ball_arr.shape[1]
P = 3
NO_OF_ITERATIONS = 9
NO_OF_BITS_TO_CHANGE = 9000

epsilon = np.asarray([ball_arr, cat_arr, mona_arr])
random_pattern = np.random.randint(P)
arr = epsilon[random_pattern]
mask_image = np.zeros((90, 100))
mask_image[:45,20:65] = arr[:45,20:65]
test_array = mask_image
test_array = test_array.reshape(1, N)
epsilon = epsilon.reshape(3, N)

# plt.imshow(test_array.reshape(90, 100), cmap='gray')
# plt.show()
# plt.imshow(arr, cmap='gray')
# plt.show()

W = np.zeros((N,N))
for i in range(3):
  W_temp = np.outer(epsilon[i], epsilon[i])
  np.fill_diagonal(W_temp, 0)
  W += W_temp

W = W / N

X = np.ones((9000, 9000))
X[:4500,:4500] = 0
np.fill_diagonal(X, 0)

W = np.multiply(W, X)

h = np.zeros((N))
rms = np.zeros((NO_OF_ITERATIONS))
img = []

for iteration in tqdm(range(NO_OF_ITERATIONS)):
	  for i in range(N):
	      i = np.random.randint(N)
	      h[i] = 0
	      for j in range(N):
	          h[i] += W[i, j]*test_array[0,j]
	  test_array = (np.where(h<0, -1, 1)).reshape(1, N)
	  rms[iteration] = mean_squared_error(test_array, ball_arr.reshape(1, N), squared=False)
	  #rms[iteration] = mean_squared_error(test_array, cat_arr.reshape(1, N), squared=False)
	  #rms[iteration] = mean_squared_error(test_array, mona_arr.reshape(1, N), squared=False)
	  img.append(test_array.reshape(90,100))
	  plt.imshow(np.where(test_array.reshape(90,100)<0, -1, 1), cmap='gray')
	  plt.show()