# MNIST-single-layer-perceptron

<p>
  This is a single layer perceptron with 785 inputs (including bias=1) and 10 outputs that correspond to digits(0-9).
  The inputs are a flattened 28x28 matrix of colors valued 0 to 255, scaled down to be between 0 and 1. Initial 
  weights are chosen randomly with mean 0 and standard deviation 0.5. It should be easy to change the learning rate,
  max epochs, and tolerance at which to prematurely stop training in the code, below the name guard. 
</p>

<p>
  Statistics are kept for the accuracy on the train and test sets after each epoch. Epoch 0 does not change the
  weights and only collects statistics. A graph of the accuracies on the epochs will be returned with a confusion
  matrix on the test set after training.
</p>

<p>
  With learning rates (0.001, 0.01, 0.1) I noticed initial accuracies average 10% and jump to about 80% after one 
  epoch and eventually level out around 85-86% with some overfitting and within usually 5 or 6 epochs fell within a 
  training tolerance of 0.001. Running without a tolerance did not noticeably improve accuracy, but accuracies on the
  test set noticably oscillated.
</p>
