import numpy as np
# from sklearn.base import RegressorMixin,BaseEstimator


class myLinearRegression_multiD:

    def __init__(self):
        self.betas = None

    def fit(self, X, y):
        matrix_plan = np.c_[np.ones((X.shape[0], 1)), X]
        self.betas = np.linalg.inv(matrix_plan.T @ matrix_plan) @ matrix_plan.T @ y
        return self

    def predict(self, X):
        matrix_plan = np.c_[np.ones((X.shape[0], 1)), X]
        return np.ravel(matrix_plan @ self.betas)

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

# or myLinearRegression_multiD(RegressorMixin, BaseEstimator)
