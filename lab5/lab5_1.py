import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import accuracy_score
from keras.datasets import mnist  # dataset from here, 28x28 images

from lab5.lab5_models import MyTorchBinaryClassifierTrainer

# For reproducibility
torch.manual_seed(42)
np.random.seed(42)

# LOAD DATA:
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("x_train.shape = ", x_train.shape)
print("x_test.shape = ", x_test.shape)

# We consider only classes 0 and 1
x_train2 = x_train[(y_train == 0) | (y_train == 1)]
y_train2 = y_train[(y_train == 0) | (y_train == 1)]

x_test2 = x_test[(y_test == 0) | (y_test == 1)]
y_test2 = y_test[(y_test == 0) | (y_test == 1)]

# nr of samples, size  of images
n_samples_train, h, w = x_train2.shape
n_samples_test, _, _ = x_test2.shape
print("images of size: h= ", h, ", w = ", w)

# reshape nr_samples * "vectorized image"
num_features = h*w
x_train2 = x_train2.reshape((-1, h*w))
x_test2 = x_test2.reshape((-1, h*w))

print("x_train2.shape = ", x_train2.shape)
print("x_test2.shape = ", x_test2.shape)

# normalizing data by number of pixels
x_train2 = x_train2 / 255
x_test2 = x_test2 / 255


# MODELS WITH LOGIT:
model_lr_torch = nn.Sequential(
    nn.Linear(num_features, 1),
    nn.Sigmoid())

# MODELS WITH ONLY LINEAR LINK:
model_lr_torch_v2 = nn.Sequential(
  nn.Linear(num_features, 1),
)

# TRAINING MODEL model_lr_torch
# Creating trainer object for model_lr_torch
model_lr_torch_training = MyTorchBinaryClassifierTrainer(model=model_lr_torch)

# Create a TensorDataset and DataLoader for model_lr_torch (the same dataloader will be used for the second model)
model_lr_torch_training.create_dataloader(x_train2, y_train2)
dataloader = model_lr_torch_training.dataloader

# TRAINING model_lr_torch
start_time = datetime.now()
model_lr_torch_training.fit()
time_elapsed = datetime.now() - start_time
print('TRAINING TIME (hh:mm:ss.ms) {}'.format(time_elapsed))


# TRAINING MODEL model_lr_torch_v2
# Creating trainer object for model_lr_torch_v2 with previous used dataloader
model_lr_torch_v2_training = MyTorchBinaryClassifierTrainer(model=model_lr_torch_v2,
                                                            dataloader=dataloader,
                                                            loss_function=nn.BCEWithLogitsLoss())

# TRAINING model_lr_torch_v2
start_time = datetime.now()
model_lr_torch_v2_training.fit()
time_elapsed = datetime.now() - start_time
print('TRAINING TIME (hh:mm:ss.ms) {}'.format(time_elapsed))

# COMPARISON OF ACCURACY
x_test2_pred_output_probs = model_lr_torch_training.predict(x_test2)
x_test2_pred_output_probs_v2 = model_lr_torch_v2_training.predict(x_test2)

x_test2_pred_output_classes = (x_test2_pred_output_probs > 0.5)
x_test2_pred_output_classes_v2 = (nn.Sigmoid()(x_test2_pred_output_probs_v2) > 0.5)
# model_lr_torch_v2 - real predictions, therefore we use nn.Sigmoid() functions on predictions
print(x_test2_pred_output_probs_v2)

print("model_lr_torch (classification of 0 and 1 digits): accuracy  =  \t",
      accuracy_score(x_test2_pred_output_classes, y_test2))
print("model_lr_torch_v2 (classification of 0 and 1 digits): accuracy  =  \t",
      accuracy_score(x_test2_pred_output_classes_v2, y_test2))


# COMPARISON OF BCE OF MODELS (computing by hand):
# BCE = -mean( y*log(prob) + (1-y)*log(1-prob))
eps = 1e-7
y_test2_tensor = torch.tensor(y_test2, dtype=torch.float32).unsqueeze(1)  # shape (N,1)

BCE_torch = -torch.mean(
    y_test2_tensor * torch.log(x_test2_pred_output_probs + eps) +
    (1 - y_test2_tensor) * torch.log(1 - x_test2_pred_output_probs + eps)
)

BCE_torch_v2 = -torch.mean(
    y_test2_tensor * torch.log(nn.Sigmoid()(x_test2_pred_output_probs_v2) + eps) +
    (1 - y_test2_tensor) * torch.log(1 - nn.Sigmoid()(x_test2_pred_output_probs_v2) + eps)
)
print("PyTorch BCE on x_test2 - model_lr_torch = ", BCE_torch.item())
print("PyTorch BCE on x_test2 - model_lr_torch_v2 = ", BCE_torch_v2.item())

# PLOT OF MODELS COEFFICIENTS AND BIAS:
nn_coef = model_lr_torch[0].weight.detach().cpu().numpy().reshape(-1)
nn_bias = model_lr_torch[0].bias.detach().cpu().numpy()[0]

nn_coef_v2 = model_lr_torch_v2[0].weight.detach().cpu().numpy().reshape(-1)
nn_bias_v2 = model_lr_torch_v2[0].bias.detach().cpu().numpy()[0]

plt.figure(figsize=(14, 6), dpi=80, facecolor='w', edgecolor='k')
plt.plot(np.arange(h*w), nn_coef, label="NN coef")
plt.plot(np.arange(h*w), nn_coef_v2, label="NN_v2 coef")
plt.title("NN bias="+str(np.round(nn_bias, 3)) + ",  NN v2 bias="+str(np.round(nn_bias_v2, 3)))

print("Sum(|nn_coef|) LC: ", np.sum(np.abs(nn_coef)))
print("Sum(|nn_coef_v2|) LC: ", np.sum(np.abs(nn_coef_v2)))

plt.legend()
plt.show()

# PLOT OF MODELS COEFFICIENTS IN GRAY SCALE:
f = plt.figure(figsize=(10, 10))

ax1 = f.add_subplot(1, 2, 1)
ax1.imshow(nn_coef.reshape(h, w), cmap='gray')
ax1.set_title("model_lr_torch coefficients")

ax2 = f.add_subplot(1, 2, 2)
ax2.imshow(nn_coef_v2.reshape(h, w), cmap='gray')
ax2.set_title("model_lr_torch_v2 coefficients")

plt.show()
