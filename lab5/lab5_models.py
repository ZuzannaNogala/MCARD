import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class MyTorchClassifierTrainer:

    def __init__(self, model, dataloader=None, lr=0.01, num_epochs=150, loss_function=nn.BCELoss()):
        self.model = model
        self.dataloader = dataloader
        self.num_epochs = num_epochs
        self.lr = lr
        self.loss_history = []
        self.loss_function = loss_function

    def create_dataloader(self, x_train, y_train, batch_size=128):
        x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

        dataset = TensorDataset(x_train_tensor, y_train_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.dataloader = dataloader

    def fit(self, verbose=True):
        print(f"TRAINING MODEL... ")

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        loss_history = []

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0

            for batch_X, batch_y in self.dataloader:
                # Zero the gradients
                optimizer.zero_grad()
                # Forward pass
                output_probs = self.model(batch_X)

                loss = self.loss_function(output_probs, batch_y)

                # Backward pass and update
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            average_loss = epoch_loss / len(self.dataloader)
            loss_history.append(average_loss)
            if verbose:
                if (epoch + 1) % 20 == 0:
                    print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {average_loss:.6f}')

    def predict(self, x_test):
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        return self.model(x_test_tensor)


def categorical_crossentropy(predictions, targets, eps=1e-7):
    # predictions are probabilities, targets are one-hot encoded
    return -torch.mean(torch.sum(targets * torch.log(predictions + eps), dim=1))


class MyTwoLayerNNClassifier:

    def __init__(self, layers_sizes, n_classes, lr=0.001, n_epochs=20, batch_size=16):
        self.lr = lr
        self.batch_size = batch_size
        self.layers_sizes = layers_sizes
        self.n_classes = n_classes
        self.n_epochs = n_epochs
        self.W1 = None
        self.W2 = None

    def forward(self, x_batch):
        H = torch.matmul(x_batch, self.W1)  # 16x785, 785x128
        H_relu = torch.relu(H)

        # Augment H_relu by adding a column of ones at the beginning (for bias)
        ones = torch.ones(H_relu.shape[0], 1)  # 16 x 129
        H_aug = torch.cat((ones, H_relu), dim=1)

        y_logis = torch.matmul(H_aug, self.W2)  # 16 x 129, 129 x 4
        y_pred = torch.softmax(y_logis, dim=1)

        return y_pred

    def fit(self, x_train, y_train, verbose=True):
        x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)

        ones_column = torch.ones(x_train_tensor.size(0), 1)

        x_train_tensor_augmented = torch.cat((ones_column, x_train_tensor), dim=1)
        y_train_one_hot = F.one_hot(y_train_tensor, num_classes=self.n_classes).float()

        train_dataset_augmented = TensorDataset(x_train_tensor_augmented, y_train_one_hot)
        train_loader_augmented = DataLoader(train_dataset_augmented, batch_size=self.batch_size, shuffle=True)

        # W1 = torch.randn(x_train_tensor_augmented.size(0), self.layers_sizes[0], requires_grad=True)
        # W2 = torch.randn(self.layers_sizes[0] + 1, self.n_classes, requires_grad=True)

        self.W1 = torch.randn(x_train_tensor.size(1) + 1, self.layers_sizes[0], requires_grad=True)
        self.W2 = torch.randn(self.layers_sizes[0] + 1, self.n_classes, requires_grad=True)

        optimizer = optim.Adam([self.W1, self.W2], lr=self.lr)

        print(f"TRAINING MODEL... ")

        for epoch in range(self.n_epochs):
            running_loss = 0.0

            for batch_x, batch_y in train_loader_augmented:
                optimizer.zero_grad()
                preds_batch = self.forward(batch_x)
                loss = categorical_crossentropy(preds_batch, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * batch_x.size(0)
            epoch_loss = running_loss / len(train_dataset_augmented)

            if verbose:
                print(f"Epoch {epoch + 1}/{self.n_epochs}, Loss: {epoch_loss:.6f}")

    def predict(self, x_test):
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        x_test_tensor_aug = torch.cat((torch.ones(x_test_tensor.size(0), 1), x_test_tensor), dim=1)

        preds_test = self.forward(x_test_tensor_aug)
        predicted_classes = preds_test.argmax(dim=1)
        return predicted_classes

