import sys
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot as plt

class Perceptron:
	def __init__(self, input_size=784, output_size=10, learning_rate=0.1):
		self.input_size = input_size
		self.output_size = output_size
		self.learning_rate = learning_rate
		# output by input array with
		# mean 0 and standard deviation 0.5
		self.weights = 0.05 * np.random.randn(output_size, input_size + 1)
		self.accuracies = np.array(70)

	def activation(self, y):
		if y <= 0:
			return 0
		else:
			return 1

	def fix_weights(self, target, x, outputs):
		# if this output was the chosen prediction t=1
		# otherwise t=0
		for w, i in zip(self.weights, range(self.output_size)):
			# w is an array of 785 
			# x is an array of 785, bias at index 0
			t = 1 if i == target else 0
			y = self.activation(outputs[i])
			for wi, xi in zip(w, x):
				wi = wi - self.learning_rate * (t - y) * xi

	def predict(self, x, outputs):
		prediction = 0
		# each of the 10 outputs corresponds to a value (0-9)
		for i in range(self.output_size):
			outputs[i] = np.dot(x, self.weights[i])
			prediction = i if outputs[i] > outputs[prediction] else prediction
		return prediction

	def train(self, input_x, input_y, epochs=2):
		for i in range(epochs):
			print("epoch", i+1)
			# create tuples of inputs and their labels
			for x, target, j in zip(input_x, input_y, range(785)):
				# x is 28 x 28 matrix and t is the target scalar (0-9)
				# flatten the inputs (28 x 28) to a single dimension
				x_flat = x.flatten()
				# add bias input (1.0)
				x_flat = np.insert(x_flat, 0, 1.0)
				# get the prediction and outputs
				outputs = np.zeros(self.output_size)
				prediction = self.predict(x_flat, outputs)
				# validate prediction
				if prediction == target:
					# correct prediction
					print(j, prediction, target, "good")
				else:
					# wrong prediction
					print(j, prediction, target)
					self.fix_weights(target, x_flat, outputs)

	def save_weights(self, filepath):
		with open(filepath, 'wb') as f:
			pickle.dump(self.weights, f)

	def load_weights(self, filepath):
		with open(filepath, 'rb') as f:
			self.weights = pickle.load()

def main() -> int:
	# load the MNIST dataset
	mnist = tf.keras.datasets.mnist
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	# plt.matshow(x_train[0])

	# Normalize the data (Scale data to be between 0 and 1)
	x_train, x_test = x_train / 255.0, y_train / 255.0

	# create 3 perceptrons with learning rates eta = 0.001, 0.01, 0.1
	#perceptron1 = Perceptron(learning_rate=0.001)
	#perceptron2 = Perceptron(learning_rate=0.01)
	perceptron3 = Perceptron(learning_rate=0.1)

	#perceptron1.train(x_train, y_train)
	#perceptron2.train(x_train, y_train)
	perceptron3.train(x_train, y_train)

	# save perceptron training weights
	#perceptron1.save_weights('p1_weights.pkl')
	#perceptron2.save_weights('p2_weights.pkl')
	#perceptron3.save_weights('p3_weights.pkl')

	return 0

if __name__ == '__main__':
	sys.exit(main())
