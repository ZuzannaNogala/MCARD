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

# PCA 3d

model_PCA = PCA(n_components=3)
model_PCA.fit(x_train)
points_train_PCA = model_PCA.transform(x_train)
points_reconstructed = model_PCA.inverse_transform(points_train_PCA)

# Errors:

AE_torch_err = np.sum((x_train-x_train_torch_reconstr.detach().numpy())**2)/(n_samples_train*image_size)
PCA_err = np.sum((x_train-points_reconstructed)**2)/(n_samples_train*image_size)

print(f"PCA reconstruction error: {np.round(PCA_err, 3)}")
print(f"Torch reconstruction error: {np.round(AE_torch_err, 3)}")

# Visualisations

fig = plt.figure(figsize=(9, 5))
ax_PCA = fig.add_subplot(1, 2, 1, projection='3d')
ax_PCA.set_title("PCA")

for label in range(len(np.unique(y_train))):
    points_PCA = points_train_PCA[y_train == label]
    ax_PCA.scatter(points_PCA[:, 0], points_PCA[:, 1], points_PCA[:, 2])

ax_2 = fig.add_subplot(1, 2, 2, projection='3d')
ax_2.plot([1, 2], [4, 1])

for label in range(len(np.unique(y_train))):
    points_torch = x_train_torch_encoded[y_train == label]
    ax_2.scatter(points_torch[:, 0], points_torch[:, 1], points_torch[:, 2])

ax_2.set_title("Pytorch")
plt.show()

toPlot = (x_train, points_reconstructed, x_train_torch_reconstr.detach().numpy())
text_rows = ["orig img", "PCA", "Pythorch"]

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
