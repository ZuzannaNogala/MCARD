import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing


data = fetch_california_housing()
X = data.data      # Features: various housing attributes
y = data.target    # Target: house price (in 100,000's)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_prediction_full = linear_model.predict(X_test)
print("MSE of full model is:", mean_squared_error(y_test, y_prediction_full))


def find_index_of_m_largest_coefs(m, model):
    coefs = model.coef_
    sorted_ind = np.array(abs(coefs)).argsort()
    return sorted_ind[-m:]


for m in [2, 4, 6]:
    vars_inds = find_index_of_m_largest_coefs(m, linear_model)

    X_train_m_vars = X_train[:, vars_inds]
    X_test_m_vars = X_test[:, vars_inds]

    model_m_vars = LinearRegression()
    y_prediction_m_vars = model_m_vars.fit(X_train_m_vars, y_train).predict(X_test_m_vars)

    print(f"MSE of model with {m} variables with larger influence is:", mean_squared_error(y_test, y_prediction_m_vars))
