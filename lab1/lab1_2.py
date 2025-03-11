from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from models import myLinearRegression_multiD
import matplotlib.pyplot as plt

data = fetch_california_housing()
X = data.data      # Features: various housing attributes
y = data.target    # Target: house price (in 100,000's)

# Q1.2 a

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

linear_model = myLinearRegression_multiD()
y_prediction = linear_model.fit(X_train, y_train).predict(X_test)
mse = mean_squared_error(y_test, y_prediction)
print("MSE: ", mse)

# Q1.2 b
model_kcv = myLinearRegression_multiD()

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model_kcv, X, y, cv=kf, scoring='neg_mean_squared_error')
print("Average MSE: ", -cv_scores.mean())


# Q1.2 c

plt.scatter(y_test, y_prediction, alpha=0.7)
plt.xlabel("Actual house price")
plt.ylabel("Predicted house price")
plt.title("Actual vs predicted house prices on the test set")
plt.show()

# Q1.2 d

plt.hist(y_prediction - y_test, bins=50, density=True)
plt.xlabel("Model residuals")
plt.xlim(-4, 4)
plt.title(r"Histogram of model residuals ($\hat{y} - y$)")
plt.show()

# the distribution of errors is approximately normal 
