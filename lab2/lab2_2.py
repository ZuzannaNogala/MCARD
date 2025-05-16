from scipy.stats import uniform
from sklearn.datasets import fetch_california_housing
import numpy as np
from sklearn.metrics import mean_squared_error
from lab2.models import myRidgeRegression_multiD, TorchLinearRegression1D
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from lab1.models import myLinearRegression_multiD

# Setup
nr_points = 200
np.random.seed(42)
x_train = np.random.rand(nr_points, 1)

a_true = 2
b_true = 1

y_train = b_true + a_true * x_train + .1 * np.random.randn(nr_points, 1)

x_test = np.random.rand(50, 1)
y_test = b_true + a_true * x_test + .1 * np.random.randn(50, 1)

# California Housing Prices data set

data = fetch_california_housing()
X = data.data
Y = data.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Q2 a

model_ridge_reg = myRidgeRegression_multiD(lr=0.01, lmb=0.0001, n_epochs=1000, optimizer_name="sgd")
model_ridge_reg.fit(x_train, y_train)
print(model_ridge_reg.params)

model_2 = TorchLinearRegression1D(lr=0.01, lmb=0.0001, n_epochs=1000)
model_2.fit(x_train, y_train)
print(model_2.b, model_2.a)


# Q2 b

lambda_grid = {'lmb': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 2]}

model = myRidgeRegression_multiD(lr=0.01, n_epochs=1000)
grid_search = GridSearchCV(model, lambda_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, Y_train)
y_hat = grid_search.best_estimator_.predict(X_test)

print(grid_search.cv_results_.get('mean_test_score'))
grid_search.cv_results_.get()

print("GridSearchCV best lambda:", grid_search.best_params_['lmb'])
print("GridSearchCV best CV MSE:", -grid_search.best_score_)
print("GridSearchCV best CV MSE (test set):", mean_squared_error(Y_test, y_hat))


model_linear_reg = myLinearRegression_multiD()
Y_prediction = model_linear_reg.predict(X_test)

print(f"MSE of linear model: {mean_squared_error(Y_test, Y_prediction)}")

# Q2 c
lambda_dist = {'lmb': uniform(0, 5)}
model_3 = myRidgeRegression_multiD()

random_search = RandomizedSearchCV(model_3, lambda_dist, cv=5, scoring='neg_mean_squared_error',
                                   n_iter=100, random_state=42)
random_search.fit(X_train, Y_train)

print("RandomizedSearchCV best lambda:", random_search.best_params_['lmb'])
print("RandomizedSearchCV best CV MSE:", -random_search.best_score_)

Y_hat = random_search.predict(X_test)

print("RandomizedSearchCV best CV MSE (test set):", mean_squared_error(Y_test, Y_hat))
