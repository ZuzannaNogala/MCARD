import numpy as np
from lab2.models import TorchLinearRegression1D, myRidgeRegression_multiD
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import torch

# Setup
nr_points = 200
np.random.seed(42)

x_train = np.random.rand(nr_points, 1)

a_true = 2.0
b_true = 1.0

y_train = b_true + a_true * x_train + .1 * np.random.randn(nr_points, 1)

x_test = np.random.rand(50, 1)
y_test = b_true + a_true * x_test + .1 * np.random.randn(50, 1)

# Q2.1

LinReg = LinearRegression()
LinReg.fit(x_train, y_train)
a_est = LinReg.coef_.item()
b_est = LinReg.intercept_.item()

loss_final = np.mean((a_est * x_test + b_est - y_test) ** 2)

print(f"Value of loss_final is {loss_final}.")


# Q2.2

model_LT = TorchLinearRegression1D(lmb=0.1, optimizer_name="sgd")
model_LT.fit(x_train, y_train)

print(f"The most optimal params a and b for g(a, b) loss function"
      f" are: a = {float(model_LT.a)}, b = {float(model_LT.b)}.")

# ok for lambda = 0.001, reps = 10000

# print(np.linalg.inv(np.c_[np.ones((x_train.shape[0], 1)), x_train].T @ np.c_[np.ones((x_train.shape[0], 1)), x_train] +
#                     0.1 * np.identity(2)) @ np.c_[np.ones((x_train.shape[0], 1)), x_train].T @ y_train)

# Q2.3

lambda_grid = {'lmb': [0.0001, 0.001, 0.01, 0.1, 0.2]}

# model = TorchLinearRegression1D(lr=0.1, n_epochs=1000)

model = myRidgeRegression_multiD(lr=0.1, n_epochs=1000)
grid_search = GridSearchCV(model, lambda_grid, cv=5, scoring='neg_mean_squared_error')
y_hat = grid_search.fit(x_train, y_train).predict(x_test)

loss_final_cv = mean_squared_error(y_test, y_hat)

# model_2 = TorchLinearRegression1D(lr=0.1, lmb=grid_search.best_params_['lmb'], n_epochs=1000, optimizer_name="sgd")
model_2 = myRidgeRegression_multiD(lr=0.1, lmb=grid_search.best_params_['lmb'], n_epochs=1000, optimizer_name="sgd")
model_2.fit(x_train, y_train)


print(f"The best lambda: {grid_search.best_params_['lmb']}, the loss final: {loss_final_cv}")
# print(f"Then the a = {model_2.a.item()}, b = {model_2.b.item()}")
print(f"Then the a = {model_2.params}")
