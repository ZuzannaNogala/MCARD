import argparse
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


def load_data():
    parser = argparse.ArgumentParser(
        description="Train LogisticRegression on processed classification task and save predictions.")
    parser.add_argument('--x_train', type=str, default="data_train_x_task.csv",
                        help="Path to train features CSV file. (default: %(default)s)")
    parser.add_argument('--y_train', type=str, default="data_train_y_task.csv",
                        help="Path to train labels CSV file. (default: %(default)s)")
    parser.add_argument('--x_test', type=str, default="data_test_x_task.csv",
                        help="Path to test features CSV file. (default: %(default)s)")
    parser.add_argument('--predictions', type=str, default="student331738_pred.csv",
                        help="Path to save predictions as CSV. (default: %(default)s)")
    args = parser.parse_args()

    X = pd.read_csv(args.x_train)
    y = pd.read_csv(args.y_train)["class"]

    return X, y


# Load data
X_train, y_train = load_data()

x_train2, x_test2, y_train2, y_test2 = train_test_split(np.array(X_train), np.array(y_train), test_size=0.25, random_state=42)

num_classes = 2  # CLASSES 0, 1
num_features = X_train.shape[1]

x_train_tensor = torch.tensor(x_train2, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train2, dtype=torch.long).ravel()

dataset = TensorDataset(x_train_tensor, y_train_tensor)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)


def fit_the_model(x_tensor, y_tensor, dataLoader, model_to_fit, n_epochs=150, lr=0.01):
    optimizer = optim.Adam(model_to_fit.parameters(), lr=lr)

    loss_history = []
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        model_to_fit.train()
        epoch_loss = 0.0

        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            output_probs = model_to_fit(batch_X)

            loss = loss_function(output_probs, batch_y)

            # Backward pass and update
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        average_loss = epoch_loss / len(dataloader)
        loss_history.append(average_loss)
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {average_loss:.6f}')

    return model_to_fit


def compute_accuracy(x_test, y_test, fitted_model):
    fitted_model.eval()
    with torch.no_grad():
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        output = fitted_model(x_test_tensor)
        predicted_classes = torch.argmax(output, dim=1).numpy()

    predicted_classes = predicted_classes.astype(int)
    print(accuracy_score(y_test2, predicted_classes))


model1 = nn.Sequential(
    nn.Linear(num_features, 10),  # num_features = 28x28 = 784
    nn.ReLU(),
    nn.Linear(10, num_classes),
)

model2 = nn.Sequential(
    nn.Linear(num_features, 15),
    torch.nn.Hardtanh(),
    nn.ELU(alpha=0.5),
    nn.Linear(15, 8),
    nn.ReLU(),
    nn.Linear(8, num_classes),
)

model3 = nn.Sequential(
    nn.Linear(num_features, 20),
    torch.nn.Hardtanh(),
    nn.ELU(alpha=0.5),
    nn.Linear(20, 10),
    torch.nn.Hardtanh(),
    nn.Linear(10, num_classes),
)


model1_fitted = fit_the_model(x_train_tensor, y_train_tensor, dataloader, model1)
print(compute_accuracy(x_test2, y_test2, model1_fitted))  # 0.8181666666666667

model2_fitted = fit_the_model(x_train_tensor, y_train_tensor, dataloader, model2)
print(compute_accuracy(x_test2, y_test2, model2_fitted))  # 0.8113333333333334

model3_fitted = fit_the_model(x_train_tensor, y_train_tensor, dataloader, model3)
print(compute_accuracy(x_test2, y_test2, model3_fitted))  # 0.806

# AdaBoost better
