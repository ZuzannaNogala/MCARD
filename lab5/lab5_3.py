import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import time
import numpy as np

from lab5.lab5_models import categorical_crossentropy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Wine dataset: 178 samples, 13 features, 3 classes.
wine = load_wine()
X = wine.data
y = wine.target

num_classes = 3

# Split into train/test (80%/20%) with stratification.
# keeping the same percentage of each class in train/test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Original y_test: {y_test}")

# Convert data to torch tensors.
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# One-hot encode target labels (for 3 classes).
y_train_tensor = F.one_hot(y_train_tensor, num_classes=3).float()
y_test_tensor = F.one_hot(y_test_tensor, num_classes=3).float()


# Model 1 - training with batch_size = 128
batch_size = 128
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


WineClassifer_torch_model = nn.Sequential(
    nn.Linear(13, 8),
    nn.ReLU(),
    nn.Linear(8, 5),
    nn.ReLU(),
    nn.Linear(5, num_classes),
    nn.Softmax(dim=1)
).to(device)

# Setup optimizer
optimizer = optim.Adam(WineClassifer_torch_model.parameters(), lr=0.01)

# Training mode
num_epochs = 500
start_time = time.time()

for epoch in range(num_epochs):
    WineClassifer_torch_model.train()
    epoch_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        predictions = WineClassifer_torch_model(batch_x)
        loss = categorical_crossentropy(predictions, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(train_loader)
    # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

print("PyTorch training took {:.2f} seconds".format(time.time() - start_time))

# Eval mode:

WineClassifer_torch_model.eval()
with torch.no_grad():
    test_outputs = WineClassifer_torch_model(X_test_tensor.to(device))
    # Get predicted classes by taking argmax
    torch_preds = test_outputs.argmax(dim=1).cpu().numpy()
    print(f"predictions of y_test: {torch_preds} (large batches)")

# Accuracy:
print("WineClassifer_torch_model with larger batches: accuracy  = \t\t", accuracy_score(torch_preds, y_test),
      " correctly cl. cases=", np.sum(torch_preds == y_test), " out of ", len(y_test))

# Model 2 - training with batch_size = 16
batch_size2 = 16
train_loader2 = DataLoader(train_dataset, batch_size=batch_size2, shuffle=True)

WineClassifer_torch_model2 = nn.Sequential(
    nn.Linear(13, 8),
    nn.ReLU(),
    nn.Linear(8, 5),
    nn.ReLU(),
    nn.Linear(5, num_classes),
    nn.Softmax(dim=1)
).to(device)

# Setup optimizer
optimizer = optim.Adam(WineClassifer_torch_model2.parameters(), lr=0.01)

# Training mode
num_epochs = 500
start_time = time.time()

for epoch in range(num_epochs):
    WineClassifer_torch_model2.train()
    epoch_loss = 0.0
    for batch_x, batch_y in train_loader2:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        predictions = WineClassifer_torch_model2(batch_x)
        loss = categorical_crossentropy(predictions, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(train_loader2)
    # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

print("PyTorch training took {:.2f} seconds".format(time.time() - start_time))

# Eval mode:

WineClassifer_torch_model2.eval()
with torch.no_grad():
    test_outputs = WineClassifer_torch_model2(X_test_tensor.to(device))
    # Get predicted classes by taking argmax
    torch_preds = test_outputs.argmax(dim=1).cpu().numpy()
    print(f"predictions of y_test: {torch_preds} (small batches)")

# Accuracy:
print("WineClassifer_torch_model with smaller batches: accuracy  = \t\t", accuracy_score(torch_preds, y_test),
      " correctly cl. cases=", np.sum(torch_preds == y_test), " out of ", len(y_test))
