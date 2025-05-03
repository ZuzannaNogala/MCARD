from matplotlib import pyplot as plt
from keras.datasets import mnist
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# READ DATA
(x_train, y_train), (x_test, y_test) = mnist.load_data()

n_samples_train, h, w = x_train.shape
n_samples_test, _, _ = x_test.shape

image_size = h*w

x_train = x_train.reshape(x_train.shape[0], 784) / 255
x_test = x_test.reshape(x_test.shape[0], 784) / 255

x_train_tensor = torch.tensor(x_train, dtype=torch.float32)

# PYTORCH LARGE
dataset = TensorDataset(x_train_tensor, x_train_tensor)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

torch_large_encoder = torch.nn.Sequential(
    torch.nn.Linear(784, 512),
    torch.nn.ELU(),
    torch.nn.Linear(512, 128),
    torch.nn.ELU(),
    torch.nn.Linear(128, 2),
)

torch_large_decoder = torch.nn.Sequential(
    torch.nn.Linear(2, 128),
    torch.nn.ELU(),
    torch.nn.Linear(128, 512),
    torch.nn.ELU(),
    torch.nn.Linear(512, 784),
    torch.nn.Sigmoid(),
)

AE_torch_large = torch.nn.Sequential(
    torch_large_encoder,
    torch_large_decoder
)

loss_torch = torch.nn.MSELoss()
optimizer = torch.optim.Adam(AE_torch_large.parameters())

n_epochs = 15
for epoch in range(n_epochs):
    running_loss = 0.0
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = AE_torch_large(inputs)
        loss = loss_torch(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(dataset)
    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.6f}")


x_train_torch_torch_encoded = torch_large_encoder(x_train_tensor)

x_train_torch_large_reconstr = AE_torch_large(x_train_tensor)

plt.figure(figsize=(9, 3))
toPlot = (x_train, x_train_torch_large_reconstr.detach().numpy())
for i in range(10):
    for j in range(2):
        ax = plt.subplot(4, 10, 10*j+i+1)
        plt.imshow(np.array(toPlot[j][i, :]).reshape(28, 28), interpolation="nearest",
                   vmin=0, vmax=1)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

plt.show()
