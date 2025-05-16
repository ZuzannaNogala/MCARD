import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class MyNN_ModelFitting:

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



