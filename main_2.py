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

def calc_weight(epsilon):
	W = np.outer(epsilon, epsilon)
	np.fill_diagonal(W, 0)
	W = W / W.shape[0]
	return W

def main():
	#path_ball = "/content/drive/MyDrive/Computational Neuroscience/ball.txt"
	path_ball = input("Enter the path of the text file of the image: ")
	ball_arr = convert_txt_to_array(path_ball, np.zeros((90, 100)))

	mask_image = np.zeros((90, 100))
	mask_image[:35,20:55] = ball_arr[:35,20:55]
	N = ball_arr.shape[0]*ball_arr.shape[1] #No. of neurons
	NO_OF_ITERATIONS = 8
	epsilon1 = ball_arr.reshape(1, N)
	test_array = mask_image.reshape(1, N)

	W = calc_weight(epsilon1) #Weight Matrix
	h = np.zeros((N))
	rms = np.zeros((NO_OF_ITERATIONS))

	for iteration in tqdm(range(NO_OF_ITERATIONS)):
	    for i in range(N):
	        i = np.random.randint(N)
	        h[i] = 0
	        for j in range(N):
	            h[i] = W[i, j]*test_array[0, j]
	    test_array = (np.where(h<0, -1, 1)).reshape(1, N)
	    rms[iteration] = mean_squared_error(test_array, ball_arr.reshape(1, N), squared=False)
	    plt.imshow(np.where(test_array.reshape(90,100)<0, -1, 1), cmap='gray')
	    plt.show()

	#plot the root mean squared error vs time
	plt.plot(np.arange(NO_OF_ITERATIONS), rms)
	plt.xlabel('Time')
	plt.ylabel('RMS error')
	plt.title('RMS error vs Time')
	plt.show()


main()