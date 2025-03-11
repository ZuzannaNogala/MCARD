from scipy.stats import uniform
from sklearn.datasets import fetch_california_housing
import numpy as np
from lab2.models import myRidgeRegression_multiD
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

model_ridge_reg = myRidgeRegression_multiD()
model_ridge_reg.fit(x_train, y_train)

# Q2 b

lambda_grid = {'lr': [0.0001, 0.001, 0.01, 0.1, 0.2]}
grid_search = GridSearchCV(myRidgeRegression_multiD(n_epochs=100), lambda_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, Y_train)

print("GridSearchCV best lambda:", grid_search.best_params_['lr'])
print("GridSearchCV best CV MSE:", -grid_search.best_score_)

model_ridge_reg_gridsearch = myRidgeRegression_multiD(lr=grid_search.best_params_['lr'])
Y_prediction_grid = model_ridge_reg_gridsearch.fit(X_train, Y_train).predict(X_test)

model_linear_reg = myLinearRegression_multiD()
Y_prediction = model_linear_reg.fit(X_train, Y_train).predict(X_test)

# Q2 c

lambda_dist = {'lr': uniform(0, 5)}

random_search = RandomizedSearchCV(myRidgeRegression_multiD(), lambda_dist, cv=5, scoring='neg_mean_squared_error',
                                   n_iter=100, random_state=42)
random_search.fit(X_train, Y_train)

print("RandomizedSearchCV best lambda:", random_search.best_params_['lr'])
print("RandomizedSearchCV best CV MSE:", -random_search.best_score_)

model_ridge_reg_randomsearch = myRidgeRegression_multiD(lr=random_search.best_params_['lr'])
Y_prediction_random = model_ridge_reg_randomsearch.fit(X_train, Y_train).predict(X_test)

print(f"MSE on test set of ridge regression model with chosen lambda is: {np.mean((Y_prediction_grid - Y_test) ** 2)}.")
print(f"MSE on test set of ridge regression model with RandomizedSearch lambda is: "
      f"{np.mean((Y_prediction_grid - Y_test) ** 2)}.")
print(f"MSE on test set of classic linear model is: {np.mean((Y_prediction - Y_test) ** 2)}.")

