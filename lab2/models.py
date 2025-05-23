import numpy as np
import torch
from sklearn.base import RegressorMixin, BaseEstimator


def greater_than_half(Matrix):
    with torch.no_grad():
        elements_to_fix = Matrix < 0.5
        Matrix[elements_to_fix] = 0.501  # In-place modification without creating new tensors
    return Matrix


class myRidgeRegression_multiD(RegressorMixin, BaseEstimator):

    def __init__(self, lr=0.06, lmb=0, n_epochs=500, optimizer_name="Adam", device=None):

        self.lr = lr
        self.lmb = lmb
        self.n_epochs = n_epochs
        self.optimizer_name = optimizer_name
        self.device = device if device is not None else torch.device("cpu")
        self.params = None
        self.loss_history = []
        self.params_history = []

    def fit(self, x_train, y_train, verbose=False):

        x_train_tensor = torch.tensor(np.c_[np.ones((x_train.shape[0], 1)), x_train], dtype=torch.float32,
                                      device=self.device).view(x_train.shape[0], x_train.shape[1] + 1)

        y_train_tensor = torch.tensor(np.ravel(y_train), dtype=torch.float32, device=self.device).view(-1, 1)

        self.params = torch.nn.Parameter(torch.randn(x_train_tensor.shape[1], 1, dtype=torch.float32, device=self.device))

        if self.optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD([self.params], lr=self.lr)
        elif self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam([self.params], lr=self.lr)
        else:
            raise ValueError("Unsupported optimizer. Choose 'SGD' or 'Adam'.")

        # Reset histories
        self.loss_history = []
        self.params_history = []

        # Main training loop
        for epoch in range(self.n_epochs):
            yhat = torch.matmul(x_train_tensor, self.params)
            error = y_train_tensor - yhat
            loss = torch.mean(torch.pow(error, 2)) + self.lmb * torch.sum(torch.pow(self.params, 2))

            # Backward pass: compute gradients.
            loss.backward()

            # Update parameters.
            optimizer.step()
            optimizer.zero_grad()

            # Record training history every 10 epochs.
            if epoch % 10 == 0:
                self.loss_history.append(loss.item())
                self.params_history.append([self.params[i] for i in range(len(self.params))])
                if verbose:
                    print(f"Epoch {epoch}: loss = {loss.item():.4f}")

        return self

    def predict(self, x):
        """
        Predict target values for new data.

        Parameters:
            x (array-like): 1D array of input features.

        Returns:
            np.ndarray: Predicted target values.
        """

        x_tensor = torch.tensor(np.c_[np.ones((x.shape[0], 1)), x], dtype=torch.float32,
                                device=self.device).view(x.shape[0], x.shape[1] + 1)
        y_prediction = x_tensor @ self.params
        return y_prediction.detach().cpu().numpy().flatten()


class TorchLinearRegression1D(RegressorMixin, BaseEstimator):
    """
    A 1D linear regression model using PyTorch and gradient descent.

    Model: y = a * x + b

    This class provides .fit() and .predict() methods, and stores the training history.
    """

    def __init__(self, lr=0.06, lmb=0, n_epochs=500, optimizer_name="Adam", device=None):
        """
        Initialize the model with hyperparameters.

        Parameters:
            lr (float): Learning rate.
            lmb (float): penalty value; if lmb = 0, then the standard linear regression is produce
            n_epochs (int): Number of training epochs.
            optimizer_name (str): Optimizer to use ("SGD" or "Adam").
            device (torch.device): Device to run computations on. Defaults to CPU.
        """
        self.lr = lr
        self.lmb = lmb
        self.n_epochs = n_epochs
        self.optimizer_name = optimizer_name
        self.device = device if device is not None else torch.device("cpu")
        self.a = None
        self.b = None
        self.loss_history = []
        self.a_history = []
        self.b_history = []

    def fit(self, x_train, y_train, verbose=False):
        """
        Train the model using gradient descent.

        Parameters:
            x_train (array-like): 1D array of input features.
            y_train (array-like): 1D array of target values.
            verbose (bool): If True, print loss every 10 epochs.

        Returns:
            self: Fitted estimator.
        """
        # Convert training data to torch tensors and ensure they are column vectors.
        x_train_tensor = torch.tensor(np.ravel(x_train), dtype=torch.float32, device=self.device).view(-1, 1)
        y_train_tensor = torch.tensor(np.ravel(y_train), dtype=torch.float32, device=self.device).view(-1, 1)

        # Initialize parameters a and b randomly, with gradient tracking.
        self.a = torch.randn(1, requires_grad=True, dtype=torch.float32, device=self.device)
        self.b = torch.randn(1, requires_grad=True, dtype=torch.float32, device=self.device)

        # Choose optimizer
        if self.optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD([self.a, self.b], lr=self.lr)
        elif self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam([self.a, self.b], lr=self.lr)
        else:
            raise ValueError("Unsupported optimizer. Choose 'SGD' or 'Adam'.")

        # Reset histories
        self.loss_history = []
        self.a_history = []
        self.b_history = []

        # Main training loop
        for epoch in range(self.n_epochs):
            # Forward pass: compute predictions and loss.
            yhat = self.a * x_train_tensor + self.b
            error = y_train_tensor - yhat
            loss = torch.mean(error ** 2) + self.lmb * (self.a ** 2 + self.b ** 2)

            # Backward pass: compute gradients.
            loss.backward()

            # Update parameters.
            optimizer.step()
            optimizer.zero_grad()

            # Record training history every 10 epochs.
            if epoch % 10 == 0:
                self.loss_history.append(loss.item())
                self.a_history.append(self.a.item())
                self.b_history.append(self.b.item())
                if verbose:
                    print(f"Epoch {epoch}: loss = {loss.item():.4f}")

        return self

    def predict(self, x):
        """
        Predict target values for new data.

        Parameters:
            x (array-like): 1D array of input features.

        Returns:
            np.ndarray: Predicted target values.
        """
        x_tensor = torch.tensor(np.ravel(x), dtype=torch.float32, device=self.device).view(-1, 1)
        y_prediction = self.a * x_tensor + self.b
        return y_prediction.detach().cpu().numpy().flatten()


class Recovering:

    def __init__(self, lr=0.06, n_epochs=500, optimizer_name="Adam", device=None):
        self.lr = lr
        self.n_epochs = n_epochs
        self.optimizer_name = optimizer_name
        self.W_r = None
        self.H_r = None
        self.device = device if device is not None else torch.device("cpu")
        self.loss_list = []

    def fit(self, Z, r, dist_pow, verbose=True):
        self.loss_list = []

        W_r = torch.randn((Z.shape[0], r), requires_grad=True, dtype=torch.float, device=self.device)
        H_r = torch.randn((r, Z.shape[1]), requires_grad=True, dtype=torch.float, device=self.device)

        if self.optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD([W_r, H_r], lr=self.lr)
        elif self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam([W_r, H_r], lr=self.lr)
        else:
            raise ValueError("Unsupported optimizer. Choose 'SGD' or 'Adam'.")

        for epoch in range(self.n_epochs):
            loss = torch.mean(torch.pow(Z - torch.matmul(W_r, H_r), dist_pow))

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            if epoch % 10 == 0:
                self.loss_list.append(loss.item())
                if verbose:
                    print(f"Epoch {epoch}: loss = {loss.item():.4f}")

        self.W_r = W_r
        self.H_r = H_r

    def fit_H_greater_than_half(self, Z, r, dist_pow, verbose=True):
        self.loss_list = []

        W_r = torch.randn((Z.shape[0], r), requires_grad=True, dtype=torch.float, device=self.device)
        H_r = torch.randn((r, Z.shape[1]), requires_grad=True, dtype=torch.float, device=self.device)

        if self.optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD([W_r, H_r], lr=self.lr)
        elif self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam([W_r, H_r], lr=self.lr)
        else:
            raise ValueError("Unsupported optimizer. Choose 'SGD' or 'Adam'.")

        for epoch in range(self.n_epochs):
            H2_r = greater_than_half(H_r)

            loss = torch.mean(torch.pow(Z - torch.matmul(W_r, H2_r), dist_pow))

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            if epoch % 10 == 0:
                self.loss_list.append(loss.item())
                if verbose:
                    print(f"Epoch {epoch}: loss = {loss.item():.4f}")

        self.W_r = W_r
        self.H_r = greater_than_half(H_r)

    def fit_nonnegativeW(self, Z, r, dist_pow, verbose=True):
        self.loss_list = []

        W_r = torch.randn((Z.shape[0], r), requires_grad=True, dtype=torch.float, device=self.device)
        H_r = torch.randn((r, Z.shape[1]), requires_grad=True, dtype=torch.float, device=self.device)

        if self.optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD([W_r, H_r], lr=self.lr)
        elif self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam([W_r, H_r], lr=self.lr)
        else:
            raise ValueError("Unsupported optimizer. Choose 'SGD' or 'Adam'.")

        for epoch in range(self.n_epochs):
            W2_r = torch.exp(W_r)

            loss = torch.mean(torch.pow(Z - torch.matmul(W2_r, H_r), dist_pow))

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            if epoch % 10 == 0:
                self.loss_list.append(loss.item())
                if verbose:
                    print(f"Epoch {epoch}: loss = {loss.item():.4f}")

        self.W_r = torch.exp(W_r)
        self.H_r = H_r

    def get_recovered_Z(self):
        return torch.matmul(self.W_r, self.H_r)


