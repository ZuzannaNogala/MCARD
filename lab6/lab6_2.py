import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from lab6.lab6_models import MyBagging


def generate_circle_data(n_pos, n_neg):
    r_pos = np.random.uniform(0, 1, size=n_pos) ** 0.5  # sqrt to ensure uniform density
    theta_pos = np.random.uniform(0, 2*np.pi, size=n_pos)
    x_pos = np.stack([r_pos * np.cos(theta_pos), r_pos * np.sin(theta_pos)], axis=1)
    y_pos = np.ones(n_pos)

    r_neg = np.random.uniform(1, 3, size=n_neg) ** 0.5
    theta_neg = np.random.uniform(0, 2*np.pi, size=n_neg)
    x_neg = np.stack([r_neg * np.cos(theta_neg), r_neg * np.sin(theta_neg)], axis=1)
    y_neg = -np.ones(n_neg)

    # Combine
    X = np.vstack([x_pos, x_neg])
    y = np.concatenate([y_pos, y_neg])

    return X, y


# Generate and split
X_all, y_all_11 = generate_circle_data(n_pos=400, n_neg=500)
x_train, x_test, y_train_11, y_test_11 = train_test_split(X_all, y_all_11, test_size=0.3, random_state=1)

# Create labels in {0,1} for scikit-learn classifiers
y_train01 = (y_train_11 + 1) // 2
y_test01 = (y_test_11 + 1) // 2

# Bagging
Bagging = MyBagging(num_classifier=100, max_depth=3)
Bagging.do_bagging(x_train, y_train_11)

print(f"The accuracy of bagged learner is: {Bagging.compute_accuracy(x_test, y_test_11)}.")

# AdaBoost

clf_Ada_trees = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3), n_estimators=100)
clf_Ada_trees.fit(x_train, y_train_11)
predictions = clf_Ada_trees.predict(x_test)

print(f"The accuracy of bagged learner is: {accuracy_score(predictions, y_test_11)}.")

# Plots

x_min, x_max = x_train[:, 0].min() - 0.5, x_train[:, 0].max() + 0.5
y_min, y_max = x_train[:, 1].min() - 0.5, x_train[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))


def plot_boundaries(x, y, ax, tree_classificator):
    Z_dec = tree_classificator.predict(np.c_[x.ravel(), y.ravel()])
    Z_dec = Z_dec.reshape(x.shape)
    ax.contourf(x, y, Z_dec, alpha=0.26)


f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 8))
f.suptitle("Comparison of classifiers: Bagging vs AdaBoost", fontsize=16)

ax1.scatter(x_train[y_train01 == 0][:, 0], x_train[y_train01 == 0][:, 1], c='orange', label='class 0')
ax1.scatter(x_train[y_train01 == 1][:, 0], x_train[y_train01 == 1][:, 1], c='blue', label='class 1')
ax1.set_title('Bagging')
plot_boundaries(xx, yy, ax1, Bagging)
ax1.legend()
ax1.grid(True)

ax2.scatter(x_train[y_train01 == 0][:, 0], x_train[y_train01 == 0][:, 1], c='orange', label='class 0')
ax2.scatter(x_train[y_train01 == 1][:, 0], x_train[y_train01 == 1][:, 1], c='blue', label='class 1')
ax2.set_title('AdaBoost')
plot_boundaries(xx, yy, ax2, clf_Ada_trees)
ax2.legend()
ax2.grid(True)

plt.show()
