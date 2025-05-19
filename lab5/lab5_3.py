import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import time
import numpy as np

from lab5.lab5_models import categorical_crossentropy, MyTorchClassifierTrainer


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

# Functions:


def create_dataloader(dataset, batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def accuracy_and_predictions_model(model, x_test_tensor=X_test_tensor, Y_test=y_test):
    model.eval()
    with torch.no_grad():
        test_outputs = model(x_test_tensor.to(device))
        # Get predicted classes by taking argmax
        torch_preds = test_outputs.argmax(dim=1).cpu().numpy()
        print(f"predictions of y_test: {torch_preds}")

    # Accuracy:
    print("accuracy  = \t\t", accuracy_score(torch_preds, Y_test),
          "correctly cl. cases=", np.sum(torch_preds == Y_test), " out of ", len(Y_test))

    return accuracy_score(torch_preds, Y_test)


# Model 1 - training with batch_size = 128

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = create_dataloader(train_dataset, batch_size=128)


WineClassifer_torch_model = nn.Sequential(
    nn.Linear(13, 8),
    nn.ReLU(),
    nn.Linear(8, 5),
    nn.ReLU(),
    nn.Linear(5, num_classes),
    nn.Softmax(dim=1)
).to(device)
print(" ")
print("MODEL 1 (batch_size=128)")

training1 = MyTorchClassifierTrainer(model=WineClassifer_torch_model,
                                     dataloader=train_loader,
                                     num_epochs=500,
                                     loss_function=categorical_crossentropy)

start_time = time.time()
training1.fit(verbose=True)
print("PyTorch training took {:.2f} seconds".format(time.time() - start_time))

accuracy_and_predictions_model(WineClassifer_torch_model)

# Model 2 - training with batch_size = 16

train_loader2 = create_dataloader(train_dataset, batch_size=16)

WineClassifer_torch_model2 = nn.Sequential(
    nn.Linear(13, 8),
    nn.ReLU(),
    nn.Linear(8, 5),
    nn.ReLU(),
    nn.Linear(5, num_classes),
    nn.Softmax(dim=1)
).to(device)

print(" ")
print("MODEL 2 (batch_size=16)")

training2 = MyTorchClassifierTrainer(model=WineClassifer_torch_model2,
                                     dataloader=train_loader2,
                                     num_epochs=500,
                                     loss_function=categorical_crossentropy)

start_time = time.time()
training2.fit(verbose=True)
print("PyTorch training took {:.2f} seconds".format(time.time() - start_time))

accuracy_and_predictions_model(WineClassifer_torch_model2)


# Average results:

K = 20
acc_results = np.zeros((K, 2))

for i in range(K):
    print(f"Id of Iteration {i}:")

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = create_dataloader(train_dataset, batch_size=128)

    WineClassifer_torch_model = nn.Sequential(
        nn.Linear(13, 8),
        nn.ReLU(),
        nn.Linear(8, 5),
        nn.ReLU(),
        nn.Linear(5, num_classes),
        nn.Softmax(dim=1)
    ).to(device)
    print(" ")
    print("MODEL 1 (batch_size=128)")

    training1 = MyTorchClassifierTrainer(model=WineClassifer_torch_model,
                                         dataloader=train_loader,
                                         num_epochs=500,
                                         loss_function=categorical_crossentropy)

    start_time = time.time()
    training1.fit(verbose=False)
    print("PyTorch training took {:.2f} seconds".format(time.time() - start_time))

    acc_1 = accuracy_and_predictions_model(WineClassifer_torch_model)

    # Model 2 - training with batch_size = 16

    train_loader2 = create_dataloader(train_dataset, batch_size=16)

    WineClassifer_torch_model2 = nn.Sequential(
        nn.Linear(13, 8),
        nn.ReLU(),
        nn.Linear(8, 5),
        nn.ReLU(),
        nn.Linear(5, num_classes),
        nn.Softmax(dim=1)
    ).to(device)

    print(" ")
    print("MODEL 2 (batch_size=16)")

    training2 = MyTorchClassifierTrainer(model=WineClassifer_torch_model2,
                                         dataloader=train_loader2,
                                         num_epochs=500,
                                         loss_function=categorical_crossentropy)

    start_time = time.time()
    training2.fit(verbose=False)
    print("PyTorch training took {:.2f} seconds".format(time.time() - start_time))

    acc_2 = accuracy_and_predictions_model(WineClassifer_torch_model2)

    acc_results[i, :] = [acc_1, acc_2]


print(acc_results.mean(axis=0))
