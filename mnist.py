import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def main() -> int:
	# load the MNIST dataset
	mnist = tf.keras.datasets.mnist
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	# Normalize the data (Scale data to be between 0 and 1)
	x_train, x_test = x_train / 255.0, y_train / 255.0

	plt.show()

	return 0


if __name__ == '__main__':
	sys.exit(main())
