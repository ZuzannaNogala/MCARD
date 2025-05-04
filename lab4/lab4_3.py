from matplotlib import pyplot as plt
from keras.datasets import mnist
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.decomposition import PCA
import torch
from torch.utils.data import DataLoader, TensorDataset

# READ DATA
# 60000 images 28 x 28
(x_train, y_train), (x_test, y_test) = mnist.load_data()

n_samples_train, h, w = x_train.shape
n_samples_test, _, _ = x_test.shape

image_size = h*w

# 60000 images 28 * 28 = 784 (flatten)
x_train = x_train.reshape(x_train.shape[0], 784) / 255
x_test = x_test.reshape(x_test.shape[0], 784) / 255

noise_level = 0.25

x_train_noisy = x_train + noise_level * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_level * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)

x_train_noisy_tensor = torch.tensor(x_train_noisy, dtype=torch.float32)
x_test_noisy_tensor = torch.tensor(x_test_noisy, dtype=torch.float32)

# PYTHORCH r=10 in BOOTLENECK LAYER

r = 10

dataset = TensorDataset(x_train_tensor, x_train_noisy_tensor)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

torch_large_encoder = torch.nn.Sequential(
    torch.nn.Linear(784, 512),
    torch.nn.ELU(),
    torch.nn.Linear(512, 128),
    torch.nn.ELU(),
    torch.nn.Linear(128, r),
)

torch_large_decoder = torch.nn.Sequential(
    torch.nn.Linear(r, 128),
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

n_epochs = 6
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


x_train_torch_encoded = torch_large_encoder(x_train_noisy_tensor).detach().numpy()
x_train_torch_reconstr = AE_torch_large(x_train_noisy_tensor)
x_train_torch_reconstr_np = x_train_torch_reconstr.detach().numpy()

print(x_train_torch_reconstr_np[5].shape)

j = 0
plt.figure(figsize=(10, 12))
for i in range(6):
    # Original image
    ax = plt.subplot(4, 6, i + 1)
    ax.imshow(x_train[i].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
    ax.axis('off')

    # Noisy image
    ax = plt.subplot(4, 6, i + 7)
    ax.imshow(x_train_noisy[i].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
    ax.axis('off')

    # Reconstructed image
    ax = plt.subplot(4, 6, i + 13)
    ax.imshow(np.array(x_train_torch_reconstr_np[i]).reshape(28, 28), cmap="gray", vmin=0, vmax=1)
    ax.axis('off')

plt.tight_layout()
plt.show()

x_test_reconstr = AE_torch_large(x_test_noisy_tensor).detach().numpy()
print(f"MSE test: {mean_squared_error(x_test, x_test_reconstr)}")
