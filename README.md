# MNIST-single-layer-perceptron

<p>
  This is a single layer perceptron with 785 inputs (including bias=1) and 10 outputs that correspond to digits(0-9).
  The inputs are a flattened 28x28 matrix of colors valued 0 to 255, scaled down to be between 0 and 1. Initial 
  weights are chosen randomly with mean 0 and standard deviation 0.5. It should be easy to change the learning rate,
  max epochs, and tolerance at which to prematurely stop training, in the code below the name guard. 
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

![η=0.1, tolerance=0.01](https://private-user-images.githubusercontent.com/146913704/347689379-35a7177c-06e6-4d78-b6b7-3a7a3a814880.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MjA2NzU3MDMsIm5iZiI6MTcyMDY3NTQwMywicGF0aCI6Ii8xNDY5MTM3MDQvMzQ3Njg5Mzc5LTM1YTcxNzdjLTA2ZTYtNGQ3OC1iNmI3LTNhN2EzYTgxNDg4MC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQwNzExJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MDcxMVQwNTIzMjNaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT04YjVmNzI4ZTRkNzYxODIwOGJlMWUyNjdhNTI3ZWEzZTkzM2RmNTViY2RmMmMxNmIzOTgzYWRmN2I3ZDE0YzM4JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.f0lOKmewDnTukS585esg4Lh3m74S-x92KWqcFJPGQO0)

![η=0.01, tolerance=0.0](https://private-user-images.githubusercontent.com/146913704/347690191-c5e020a6-cb90-40d2-9b8a-6db7cfe7f453.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MjA2NzU5MjcsIm5iZiI6MTcyMDY3NTYyNywicGF0aCI6Ii8xNDY5MTM3MDQvMzQ3NjkwMTkxLWM1ZTAyMGE2LWNiOTAtNDBkMi05YjhhLTZkYjdjZmU3ZjQ1My5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQwNzExJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MDcxMVQwNTI3MDdaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT00YWJiNDU2OTM4ZjBhMDg3OGU3MDE0NzE2ZTM0MDkxYTVlNTgxZjFjMGQyNzdjYmQxM2JlOGIyOGY4M2YwMTEzJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.CRYeLPMiv3XeEJur18CcE83N8EFC5D_RMLY3mxBlyKE)

![η=0.1, confusion matrix](https://private-user-images.githubusercontent.com/146913704/347689418-5e548790-5692-4960-a79a-edf4e78d8df0.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MjA2NzU3MDMsIm5iZiI6MTcyMDY3NTQwMywicGF0aCI6Ii8xNDY5MTM3MDQvMzQ3Njg5NDE4LTVlNTQ4NzkwLTU2OTItNDk2MC1hNzlhLWVkZjRlNzhkOGRmMC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQwNzExJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MDcxMVQwNTIzMjNaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0zOTk3ZWM3MjEwYjE5NjIwNzY5NWM2MTk2YTRlYTdmZDdkYTlmNzRiZGI1ODI4N2ZkMTYzYzQ1NmEyMzhkYmNmJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.9HQNgRLpUmKqXpLGIjt_PQK6HNb5uRFN2eY4xceg51k)
