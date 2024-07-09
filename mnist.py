import tensorflow as tf
import numpy as np
import pandas as pd
from mathplotlib import pyplot as plt

# load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

# Normalize the data / Scale data to be between 0 and 1
x_train, x_test = x_train / 255.0, y_train / 255.0
