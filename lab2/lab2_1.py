import numpy as np
from lab2.models import TorchLinearRegression1D
import torch

# Setup
nr_points = 200
np.random.seed(42)
x_train = np.random.rand(nr_points, 1)

a_true = 2
b_true = 1

y_train = b_true + a_true * x_train + .1 * np.random.randn(nr_points, 1)

x_test = np.random.rand(50, 1)
y_test = b_true + a_true * x_test + .1 * np.random.randn(50, 1)

# Q2.1

model_LT = TorchLinearRegression1D()
model_LT.fit(x_train, y_train)

loss_final = np.mean((float(model_LT.a) * x_test +
                      float(model_LT.b) - model_LT.predict(x_test)) ** 2)

print(f"Value of MSE of TorchLinearModel is {loss_final}.")


# Q2.2

def g_loss_fun(error, model):
    return torch.mean(error ** 2) + model.lr * (model.a ** 2 + model.b ** 2)


model_LT_gfun = TorchLinearRegression1D(lr=0.1, optimizer_name="sgd")
model_LT_gfun.fit(x_train, y_train, loss_fun=lambda error:  g_loss_fun(error, model_LT_gfun), verbose=True)

x_train_tensor = torch.from_numpy(np.c_[np.ones((x_train.shape[0], 1)), x_train]).float().to("cpu")

error_model_LT_gfun = (model_LT_gfun.a * x_train_tensor + model_LT_gfun.b -
                       torch.from_numpy(model_LT_gfun.predict(x_train)).float().to(model_LT_gfun.device))

g_final = g_loss_fun(error_model_LT_gfun, model_LT_gfun)

print(f"The most optimal params a and b for g(a, b) loss function"
      f" are: a = {float(model_LT_gfun.a)}, b = {float(model_LT_gfun.b)}.")

print(f"The value of g(a,b) function with those parameters is: {float(g_final)}.")
