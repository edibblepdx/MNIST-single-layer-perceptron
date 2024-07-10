import sys
import tensorflow as tf
import numpy as np
import pickle
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import argparse

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

	def train(self, x_train, y_train, x_test, y_test, epochs=70, tolerance=0.01):
		train_accuracies = []
		test_accuracies = []
		previous_accuracy = 0.0

		for epoch in range(epochs+1):
			num_correct = 0.0
			# create tuples of inputs and their labels
			for j in range(len(x_train)):
				# x is 28 x 28 matrix and target is a scalar (0-9)
				# flatten the inputs (28 x 28) to a single dimension
				x_flat = x_train[j].flatten()
				x_flat = np.insert(x_flat, 0, 1.0)	# add bias input (1.0)
				target = y_train[j]
				# get the prediction and outputs
				prediction, outputs = self.predict(x_flat)
				# validate prediction
				if prediction == target:
					# correct prediction
					num_correct += 1
				elif epoch != 0:
					# wrong prediction
					self.fix_weights(target, x_flat, outputs)
			
			train_accuracy = num_correct / len(x_train)
			train_accuracies.append(train_accuracy)

			test_accuracy = self.evaluate(x_test, y_test)
			test_accuracies.append(test_accuracy)

			print(f'Epoch {epoch}	: Correct Train {num_correct} : Accuracy Train {train_accuracy:.4f} : Accuracy Test {test_accuracy:.4f}')

			if abs(train_accuracy - previous_accuracy) < tolerance:
				break
			previous_accuracy = train_accuracy

		return train_accuracies, test_accuracies

	def evaluate(self, x_test, y_test):
		num_correct = 0
		for j in range(len(x_test)):
			x_flat = x_test[j].flatten()
			x_flat = np.insert(x_flat, 0, 1.0)	# add bias input (1.0)
			target = y_test[j]
			prediction, _ = self.predict(x_flat)
			if prediction == target:
				num_correct += 1
		return num_correct / len(x_test)

	def confusion_matrix(self, x, y):
		y_true = []
		y_pred = []
		for j in range(len(x)):
			x_flat = x[j].flatten()
			x_flat = np.insert(x_flat, 0, 1.0)  # Adding bias input
			target = y[j]
			prediction, _ = self.predict(x_flat)
			y_true.append(target)
			y_pred.append(prediction)
		return confusion_matrix(y_true, y_pred)

	def save_weights(self, filepath):
		with open(filepath, 'wb') as f:
			pickle.dump(self.weights, f)

	def load_weights(self, filepath):
		with open(filepath, 'rb') as f:
			self.weights = pickle.load()

def main(learning_rate, epochs, tolerance) -> int:
	# load the MNIST dataset
	mnist = tf.keras.datasets.mnist
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	# plt.matshow(x_train[0])

	# Normalize the data (Scale data to be between 0 and 1)
	x_train, x_test = x_train / 255.0, x_test / 255.0

	perceptron = Perceptron(learning_rate=learning_rate)

	train_accuracies, test_accuracies = perceptron.train(x_train, y_train, x_test, y_test, epochs=epochs, tolerance=tolerance)
	plt.plot(range(len(train_accuracies)), train_accuracies, label='train')
	plt.plot(range(len(test_accuracies)), test_accuracies, label='test')
	plt.ylim(0, 1)
	plt.xticks(range(len(train_accuracies)), range(len(train_accuracies)))
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.title(f'Perceptron Learning Accuracy (η={learning_rate})')
	plt.legend(loc='upper left')
	plt.show()

	# save perceptron training weights
	perceptron.save_weights('weights.pkl')

	# Confusion matrices for the perceptron on the test set
	cm = perceptron.confusion_matrix(x_test, y_test)
	display = ConfusionMatrixDisplay(confusion_matrix=cm)
	display.plot()
	plt.title(f'Perceptron Confusion Matrix on Test Set (η={learning_rate})')
	plt.show()

	return 0

if __name__ == '__main__':
	"""
	parser = argparse.ArgumentParser(description="Train a single-layer perceptron on the MNIST dataset.")
	parser.add_argument('-n', '--learning_rate', type=float, required=False, help='Learning rate for the perceptron.')
	parser.add_argument('-e', '--epochs', type=int, required=False, help='Number of epochs.')
	parser.add_argument('-t', '--tolerance', type=float, required=False, help='Tolerance at which to prematurely stop training.')
	args = parser.parse_args()

	learning_rate = args.learning_rate if args.learning_rate is not None else 0.001
	epochs = args.epochs if args.epochs is not None else 70
	tolerance = args.tolerance if args.tolerance is not None else 0.01
	"""
	learning_rate = 0.001
	epochs = 70
	tolerance = 0.001
	
	sys.exit(main(learning_rate, epochs, tolerance))
