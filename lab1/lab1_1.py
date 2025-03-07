import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from models import myLinearRegression_multiD

# Q1.1 a

np.random.seed(42)
n_points = 200
X_2d = np.random.rand(n_points, 2)

beta0_true = 1.0
beta1_true = 2.0
beta2_true = 3.0

y_2d = beta0_true + beta1_true * X_2d[:, 0] + beta2_true * X_2d[:, 1] + 1 * np.random.randn(n_points)

model2D = myLinearRegression_multiD()
model2D.fit(X_2d, y_2d)

print("Predicted values for two dimensional data with myLinearRegression_multiD:", model2D.predict(X_2d))

# Check if the myLinearRegression_multiD gives the same results as LinearRegression() from sklearn

# model_sklearn = LinearRegression()
# model_sklearn.fit(X_2d, y_2d)
# print(model_sklearn.predict(X_2d) - model2D.predict(X_2d))

# Q1.1 b

np.random.seed(42)
x_train = np.random.rand(n_points, 1)

a_true = 2  # slope
b_true = 1  # intercept
y_train = b_true + a_true * x_train + 0.1 * np.random.randn(n_points, 1)

model1D = myLinearRegression_multiD()
model1D.fit(x_train, y_train)

print("Predicted values for one dimensional data with myLinearRegression_multiD:", model1D.predict(x_train))

# Q1.1 c

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_2d[:, 0], X_2d[:, 1], y_2d, color='blue', alpha=0.5, label='Data points')

x1_grid, x2_grid = np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 20))
X_grid = np.column_stack((x1_grid.ravel(), x2_grid.ravel()))
y_grid = model2D.predict(X_grid).reshape(x1_grid.shape)

ax.plot_surface(x1_grid, x2_grid, y_grid, color='red', alpha=0.5)
ax.view_init(0, 0)
# plt.show()

# Q1.1 d

n_samples = 200
n_features = 100
np.random.seed(42)

X_100d = np.random.randn(n_samples, n_features)

true_intercept = 3.0
true_coef = np.random.randn(n_features)

y_100d = true_intercept + X_100d.dot(true_coef) + 0.5 * np.random.randn(n_samples)

model100D = myLinearRegression_multiD()
model100D.fit(X_100d, y_100d)

model_sklearn100D = LinearRegression()
model_sklearn100D.fit(X_100d, y_100d)

L2_norm = ((model_sklearn100D.intercept_ - model100D.betas[0]) ** 2 +
           np.sum((model_sklearn100D.coef_ - model100D.betas[1:]) ** 2))

print("The L2 norm of the difference between the recovered parameter vectors equals: ", L2_norm)
