import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.ensemble import AdaBoostClassifier

# Set seed for reproducibility
np.random.seed(42)


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

# create meshgrid:
x_min, x_max = x_train[:, 0].min() - 0.5, x_train[:, 0].max() + 0.5
y_min, y_max = x_train[:, 1].min() - 0.5, x_train[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))


def plot_boundaries(x, y, ax, tree_classificator):
    Z_dec = tree_classificator.predict(np.c_[x.ravel(), y.ravel()])
    Z_dec = Z_dec.reshape(x.shape)
    ax.contourf(x, y, Z_dec, alpha=0.26)
# cmap="bwr"


# Tree Classifiers Fitting:
clf_tree_1 = DecisionTreeClassifier(max_depth=1)
clf_tree_1.fit(x_train, y_train01)

clf_tree_2 = DecisionTreeClassifier(max_depth=2)
clf_tree_2.fit(x_train, y_train01)

clf_tree_3 = DecisionTreeClassifier(max_depth=3)
clf_tree_3.fit(x_train, y_train01)

# _ = tree.plot_tree(clf_tree_1,
#                    feature_names=["feature0","feature1"],
#                    class_names=["0","+1"],
#                    proportion=True,
#                    filled=True)
#
# _ = tree.plot_tree(clf_tree_2,
#                    feature_names=["feature0","feature1"],
#                    class_names=["0","+1"],
#                    proportion=True,
#                    filled=True)
#
# _ = tree.plot_tree(clf_tree_3,
#                    feature_names=["feature0","feature1"],
#                    class_names=["0","+1"],
#                    proportion=True,
#                    filled=True)

# Tree Classifiers Accuracy:

y_tree_1_prediction = clf_tree_1.predict(x_test)
y_tree_2_prediction = clf_tree_2.predict(x_test)
y_tree_3_prediction = clf_tree_3.predict(x_test)


print("TREE classifier (max_depth=1), ACC = ", accuracy_score(y_tree_1_prediction, y_test01))
print("TREE classifier (max_depth=2), ACC = ", accuracy_score(y_tree_2_prediction, y_test01))
print("TREE classifier (max_depth=3), ACC = ", accuracy_score(y_tree_3_prediction, y_test01))

# Tree Classifiers display:

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(10, 8))
f.suptitle("Tree classification performance with different choice of max_depth", fontsize=16)

ax1.scatter(x_train[y_train01 == 0][:, 0], x_train[y_train01 == 0][:, 1], c='orange', label='class 0')
ax1.scatter(x_train[y_train01 == 1][:, 0], x_train[y_train01 == 1][:, 1], c='blue', label='class 1')
ax1.set_title('max_depth=1')
plot_boundaries(xx, yy, ax1, clf_tree_1)
ax1.legend()
ax1.grid(True)

ax2.scatter(x_train[y_train01 == 0][:, 0], x_train[y_train01 == 0][:, 1], c='orange', label='class 0')
ax2.scatter(x_train[y_train01 == 1][:, 0], x_train[y_train01 == 1][:, 1], c='blue', label='class 1')
ax2.set_title('max_depth=2')
plot_boundaries(xx, yy, ax2, clf_tree_2)
ax2.legend()
ax2.grid(True)

ax3.scatter(x_train[y_train01 == 0][:, 0], x_train[y_train01 == 0][:, 1], c='orange', label='class 0')
ax3.scatter(x_train[y_train01 == 1][:, 0], x_train[y_train01 == 1][:, 1], c='blue', label='class 1')
ax3.set_title('max_depth=3')
plot_boundaries(xx, yy, ax3, clf_tree_3)
ax3.legend()
ax3.grid(True)

plt.show()

# Improve tree classificator with AdaBoost:

clf_Ada_1 = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=60, random_state=322)
clf_Ada_1.fit(x_train, y_train01)

clf_Ada_2 = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2), n_estimators=60, random_state=322)
clf_Ada_2.fit(x_train, y_train01)

clf_Ada_3 = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3), n_estimators=60, random_state=322)
clf_Ada_3.fit(x_train, y_train01)

# AdaBoost accuracy:

y_Ada_1_prediction = clf_Ada_1.predict(x_test)
y_Ada_2_prediction = clf_Ada_2.predict(x_test)
y_Ada_3_prediction = clf_Ada_3.predict(x_test)


print("ADA classifier on tree with max_depth=1, ACC = ", accuracy_score(y_Ada_1_prediction, y_test01))
print("ADA classifier on tree with max_depth=2, ACC = ", accuracy_score(y_Ada_2_prediction, y_test01))
print("ADA classifier on tree with max_depth=3, ACC = ", accuracy_score(y_Ada_3_prediction, y_test01))

# Display performance of AdaBoost

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(10, 8))
f.suptitle("AdaBoost performance with different tree classifier", fontsize=16)

ax1.scatter(x_train[y_train01 == 0][:, 0], x_train[y_train01 == 0][:, 1], c='orange', label='class 0')
ax1.scatter(x_train[y_train01 == 1][:, 0], x_train[y_train01 == 1][:, 1], c='blue', label='class 1')
ax1.set_title('max_depth=1')
plot_boundaries(xx, yy, ax1, clf_Ada_1)
ax1.legend()
ax1.grid(True)

ax2.scatter(x_train[y_train01 == 0][:, 0], x_train[y_train01 == 0][:, 1], c='orange', label='class 0')
ax2.scatter(x_train[y_train01 == 1][:, 0], x_train[y_train01 == 1][:, 1], c='blue', label='class 1')
ax2.set_title('max_depth=2')
plot_boundaries(xx, yy, ax2, clf_Ada_2)
ax2.legend()
ax2.grid(True)

ax3.scatter(x_train[y_train01 == 0][:, 0], x_train[y_train01 == 0][:, 1], c='orange', label='class 0')
ax3.scatter(x_train[y_train01 == 1][:, 0], x_train[y_train01 == 1][:, 1], c='blue', label='class 1')
ax3.set_title('max_depth=3')
plot_boundaries(xx, yy, ax3, clf_Ada_3)
ax3.legend()
ax3.grid(True)

plt.show()
