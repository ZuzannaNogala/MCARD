from matplotlib import pyplot as plt
from keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
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
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)

# PYTHORCH r=3 in BOOTLENECK LAYER

dataset = TensorDataset(x_train_tensor, x_train_tensor)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

torch_large_encoder = torch.nn.Sequential(
    torch.nn.Linear(784, 512),
    torch.nn.ELU(),
    torch.nn.Linear(512, 128),
    torch.nn.ELU(),
    torch.nn.Linear(128, 3),
)

torch_large_decoder = torch.nn.Sequential(
    torch.nn.Linear(3, 128),
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

n_epochs = 5
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


x_train_torch_encoded = torch_large_encoder(x_train_tensor).detach().numpy()
x_train_torch_reconstr = AE_torch_large(x_train_tensor)

# PYTHORCH r=8 in BOOTLENECK LAYER

torch_large_encoder_8 = torch.nn.Sequential(
    torch.nn.Linear(784, 512),
    torch.nn.ELU(),
    torch.nn.Linear(512, 128),
    torch.nn.ELU(),
    torch.nn.Linear(128, 8),
)

torch_large_decoder_8 = torch.nn.Sequential(
    torch.nn.Linear(8, 128),
    torch.nn.ELU(),
    torch.nn.Linear(128, 512),
    torch.nn.ELU(),
    torch.nn.Linear(512, 784),
    torch.nn.Sigmoid(),
)

AE_torch_large_8 = torch.nn.Sequential(
    torch_large_encoder_8,
    torch_large_decoder_8
)

loss_torch_8 = torch.nn.MSELoss()
optimizer_8 = torch.optim.Adam(AE_torch_large_8.parameters())

n_epochs = 5
for epoch in range(n_epochs):
    running_loss = 0.0
    for inputs, targets in dataloader:
        optimizer_8.zero_grad()
        outputs = AE_torch_large_8(inputs)
        loss = loss_torch_8(outputs, targets)
        loss.backward()
        optimizer_8.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(dataset)
    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.6f}")


x_train_torch_encoded_8 = torch_large_encoder_8(x_train_tensor).detach().numpy()
x_train_torch_reconstr_8 = AE_torch_large_8(x_train_tensor)

# PCA - 2 components

model_PCA2 = PCA(n_components=2)
model_PCA2.fit(x_train)
points_train_PCA2 = model_PCA2.transform(x_train)
points_reconstructed2 = model_PCA2.inverse_transform(points_train_PCA2)

# PCA - 3 components

model_PCA3 = PCA(n_components=3)
model_PCA3.fit(x_train)
points_train_PCA3 = model_PCA3.transform(x_train)
points_reconstructed3 = model_PCA3.inverse_transform(points_train_PCA3)

# PCA - r = 8 components

model_PCA8 = PCA(n_components=8)
model_PCA8.fit(x_train)
points_train_PCA8 = model_PCA8.transform(x_train)
points_reconstructed8 = model_PCA8.inverse_transform(points_train_PCA8)

# Errors:

AE_torch_err_3 = np.sum((x_train-x_train_torch_reconstr.detach().numpy())**2)/(n_samples_train*image_size)
AE_torch_err_8 = np.sum((x_train-x_train_torch_reconstr_8.detach().numpy())**2)/(n_samples_train*image_size)
PCA_err_8 = np.sum((x_train-points_reconstructed8)**2)/(n_samples_train*image_size)
PCA_err_3 = np.sum((x_train-points_reconstructed3)**2)/(n_samples_train*image_size)
PCA_err_2 = np.sum((x_train-points_reconstructed2)**2)/(n_samples_train*image_size)

print(f"PCA reconstruction error (r = 2): {np.round(PCA_err_2, 3)}")
print(f"PCA reconstruction error (r = 3): {np.round(PCA_err_3, 3)}")
print(f"PCA reconstruction error (r = 8): {np.round(PCA_err_8, 3)}")
print(f"Torch reconstruction error (r = 3): {np.round(AE_torch_err_3, 3)}")
print(f"Torch reconstruction error (r = 8): {np.round(AE_torch_err_8, 3)}")

# Visualisations

fig = plt.figure()
ax_PCA2 = fig.add_subplot(1, 3, 1, projection='3d')
ax_PCA2.set_title("PCA (r = 2)")
ax_PCA3 = fig.add_subplot(1, 3, 2, projection='3d')
ax_PCA3.set_title("PCA (r = 3)")

for label in range(len(np.unique(y_train))):
    points_PCA_2 = points_train_PCA2[y_train == label]
    ax_PCA2.scatter(points_PCA_2[:, 0], points_PCA_2[:, 1], s=1)

    points_PCA_3 = points_train_PCA3[y_train == label]
    ax_PCA3.scatter(points_PCA_3[:, 0], points_PCA_3[:, 1], points_PCA_3[:, 2], s=1)

ax_AE = fig.add_subplot(1, 3, 3, projection='3d')

for label in range(len(np.unique(y_train))):
    points_AE = x_train_torch_encoded[y_train == label]
    ax_AE.scatter(points_AE[:, 0], points_AE[:, 1], points_AE[:, 2], s=1)

ax_AE.set_title("Pytorch (r = 3)")
plt.show()

toPlot = (x_train, points_reconstructed2,  points_reconstructed3, points_reconstructed8,
          x_train_torch_reconstr.detach().numpy(), x_train_torch_reconstr_8.detach().numpy())
text_rows = ["orig img", "PCA r=2", "PCA r=3", "PCA r = 8", "Pytorch r=3", "Pytorch r=8"]

nRows = len(toPlot)   # Should be 3
nCols = 10            # Number of images per row

plt.figure(figsize=(9, 5))  # Increase height for 5 rows

for j in range(nRows):
    for i in range(nCols):
        ax = plt.subplot(nRows, nCols, j*nCols + i + 1)
        # Reshape each image (assumed to be flattened 28x28) and show.
        plt.imshow(np.array(toPlot[j][i, :]).reshape(28, 28), interpolation="nearest", vmin=0, vmax=1)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # Add row label on the first column of each row.
        if i == 0:
            ax.text(-0.35, 0.5, text_rows[j],
                    transform=ax.transAxes,
                    fontsize=12,
                    va='center',
                    ha='right',
                    color='red')

plt.tight_layout()
plt.show()
