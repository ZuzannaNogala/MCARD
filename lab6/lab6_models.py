import numpy as np
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


class MyBagging:

    def __init__(self, num_classifier, max_depth, random_state=None):
        self.Q = num_classifier
        self.random_state = random_state
        self.max_depth = max_depth
        self.weak_learners = []

    def create_weak_learners(self):
        self.weak_learners = [DecisionTreeClassifier(max_depth=self.max_depth) for _ in range(self.Q)]

    def do_bagging(self, x_set, y_set):
        N = len(y_set)

        self.create_weak_learners()

        for t in range(self.Q):

            if self.random_state is None:
                x_sample, y_sample = resample(x_set, y_set, replace=True,
                                              n_samples=N, random_state=1000 + t)
            else:
                x_sample, y_sample = resample(x_set, y_set, replace=True,
                                              n_samples=N, random_state=self.random_state)

            self.weak_learners[t].fit(x_sample, y_sample)

    def predict(self, x_set):
        S = np.array([1/self.Q * self.weak_learners[t].predict(x_set) for t in range(self.Q)])
        H = np.sign(np.sum(S, axis=0))
        return H

    def compute_accuracy(self, x_test, y_test):
        predictions = self.predict(x_test)
        return accuracy_score(y_test, predictions)