import sys
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot as plt

class Perceptron:
	def __init__(self, input_size=785, output_size=10, learning_rate=0.1):
		self.input_size = input_size
		self.output_size = output_size
		self.learning_rate = learning_rate
		# mean 0 and standard deviation 0.5
		self.weights = 0.05 * np.random.randn(output_size, input_size)

	def activation(self, y):
		return 1 if y > 0 else 0

	def fix_weights(self, target, x, outputs):
		for i in range(self.output_size):
			t = 1 if i == target else 0
			y = 1 if outputs[i] > 0 else 0
			for j in range(self.input_size):
				self.weights[i][j] += self.learning_rate * (t - y) * x[j]
		
	def predict(self, x):
		outputs = np.zeros(self.output_size)
		# each of the 10 outputs corresponds to a value (0-9)
		for i in range(self.output_size):
			outputs[i] = np.dot(self.weights[i], x)
			# prediction = i if outputs[i] > outputs[prediction] else prediction
		return np.argmax(outputs), outputs

	def train(self, input_x, input_y, epochs=70, tolerance=0.01):
		accuracies = []
		for i in range(epochs):
			num_correct = 0.0
			print("epoch", i+1)
			# create tuples of inputs and their labels
			for j in range(len(input_x)):
				# x is 28 x 28 matrix and target is a scalar (0-9)
				# flatten the inputs (28 x 28) to a single dimension
				x_flat = input_x[j].flatten()
				x_flat = np.insert(x_flat, 0, 1.0)	# add bias input (1.0)
				target = input_y[j]
				# get the prediction and outputs
				prediction, outputs = self.predict(x_flat)
				# validate prediction
				if prediction == target:
					# correct prediction
					num_correct += 1
				else:
					# wrong prediction
					self.fix_weights(target, x_flat, outputs)
			accuracy = num_correct / len(input_x)
			print('correct:', num_correct)
			print('accuracy:', accuracy)
			accuracies.append(accuracy)
		return accuracies

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
	x_train, x_test = x_train / 255.0, x_test / 255.0

	# create 3 perceptrons with learning rates eta = 0.001, 0.01, 0.1
	#perceptron1 = Perceptron(learning_rate=0.001)
	#perceptron2 = Perceptron(learning_rate=0.01)
	perceptron3 = Perceptron(learning_rate=0.1)

	#perceptron1.train(x_train, y_train)
	#perceptron2.train(x_train, y_train)
	accuracies = perceptron3.train(x_train, y_train)
	plt.plot(accuracies)
	plt.ylim(0, 1)
	plt.xticks(range(len(accuracies)), range(len(accuracies)))
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.show()

	# save perceptron training weights
	#perceptron1.save_weights('p1_weights.pkl')
	#perceptron2.save_weights('p2_weights.pkl')
	#perceptron3.save_weights('p3_weights.pkl')

	return 0

if __name__ == '__main__':
	sys.exit(main())
