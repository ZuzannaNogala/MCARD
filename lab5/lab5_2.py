import torch
import numpy as np
from sklearn.metrics import accuracy_score
from datetime import datetime
from keras.datasets import mnist # <-- dataset from here, 28x28 images
from lab5.lab5_models import MyTwoLayerNNClassifier

# For reproducibility
torch.manual_seed(42)
np.random.seed(42)


# LOAD DATA:
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_trainM, y_trainM = x_train, y_train
x_testM, y_testM = x_test, y_test

# CLASSES 0, 1, 2, 3
x_trainM, y_trainM = x_train[y_train <= 3], y_train[y_train <= 3]
x_testM, y_testM = x_test[y_test <= 3], y_test[y_test <= 3]


# nr of samples, size  of images
n_samples_trainM, h, w = x_trainM.shape
n_samples_testM, _, _ = x_testM.shape
print("images of size: h= ", h, ", w = ", w)

num_features = h*w

x_trainM = x_trainM.reshape((-1, h*w))
x_testM = x_testM.reshape((-1, h*w))

print("x_trainM.shape = ", x_trainM.shape)
print("x_testM.shape = ", x_testM.shape)

# normalizing
x_trainM = x_trainM / 255
x_testM = x_testM / 255


# TRAINING
model = MyTwoLayerNNClassifier(layers_sizes=[128, 4], n_classes=4, n_epochs=20)

start_time = datetime.now()
model.fit(x_trainM, y_trainM)
time_elapsed = datetime.now() - start_time
print('TRAINING TIME (hh:mm:ss.ms) {}'.format(time_elapsed))

# CHECK ACCURACY ON TEST SET
predicted_classes = model.predict(x_testM)

print("W1, W2, accuracy  = \t", accuracy_score(predicted_classes,y_testM),
      " correctly cl. cases=", np.sum(predicted_classes.detach().numpy() == y_testM))
