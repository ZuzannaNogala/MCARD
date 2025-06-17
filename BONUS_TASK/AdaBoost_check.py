import argparse
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def load_data():
    parser = argparse.ArgumentParser(
        description="Train LogisticRegression on processed classification task and save predictions.")
    parser.add_argument('--x_train', type=str, default="data_train_x_task.csv",
                        help="Path to train features CSV file. (default: %(default)s)")
    parser.add_argument('--y_train', type=str, default="data_train_y_task.csv",
                        help="Path to train labels CSV file. (default: %(default)s)")
    parser.add_argument('--x_test', type=str, default="data_test_x_task.csv",
                        help="Path to test features CSV file. (default: %(default)s)")
    parser.add_argument('--predictions', type=str, default="student331738_pred.csv",
                        help="Path to save predictions as CSV. (default: %(default)s)")
    args = parser.parse_args()

    X = pd.read_csv(args.x_train)
    y = pd.read_csv(args.y_train)["class"]

    return X, y


def grid_search_AdaBoost(params_grid, num_cv=5):

    base_estimator = DecisionTreeClassifier()
    AdaModel = AdaBoostClassifier(estimator=base_estimator)
    grid_search = GridSearchCV(AdaModel, params_grid, cv=num_cv, scoring='accuracy', verbose=2)
    grid_search.fit(X_train, y_train)

    return grid_search.best_params_, grid_search.best_score_


# Load data
X_train, y_train = load_data()

# first grid search for large learning rate
param_grid = {
    'estimator__max_depth': [1, 2, 3, 4],
    'estimator__criterion': ['gini', 'entropy'],
    'estimator__splitter': ['best', 'random'],
    'n_estimators': [50, 100, 150],
    'learning_rate': [1]
}

# print(grid_search_AdaBoost(param_grid))

# results: {'estimator__criterion': 'gini', 'estimator__max_depth': 3,
#           'estimator__splitter': 'best', 'learning_rate': 1, 'n_estimators': 50}
# Accuracy: 0.8203333333333334


# second grid search
# to find if the number of n_estimator can be better
# to find learning rate

param_grid_2 = {
    'estimator__max_depth': [3],
    'estimator__criterion': ['gini'],
    'estimator__splitter': ['best'],
    'n_estimators': [10, 20, 40, 50, 60],
    'learning_rate': [0.001, 0.01, 0.1, 0.5, 1]
}

# print(grid_search_AdaBoost(param_grid_2))

# results: {'estimator__criterion': 'gini', 'estimator__max_depth': 3,
# 'estimator__splitter': 'best', 'learning_rate': 0.01, 'n_estimators': 60},
# Accuracy: 0.8210416666666667

# third grid search - check other n_estimators grid

param_grid_3 = {
    'estimator__max_depth': [3],
    'estimator__criterion': ['gini'],
    'estimator__splitter': ['best'],
    'n_estimators': [60, 70, 80, 90, 100],
    'learning_rate': [0.1]
}

print(grid_search_AdaBoost(param_grid_3))

# results: {'estimator__criterion': 'gini', 'estimator__max_depth': 3,
# 'estimator__splitter': 'best', 'learning_rate': 0.1, 'n_estimators': 80}
# Accuracy: 0.8209583333333332

# check if n_estimators=80 is better:
x_train2, x_test2, y_train2, y_test2 = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

model = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3), n_estimators=80, learning_rate=0.1)
model2 = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3), n_estimators=60, learning_rate=0.1)

model.fit(x_train2, y_train2)
model2.fit(x_train2, y_train2)

print(f"Accuracy model: {accuracy_score(model.predict(x_test2), y_test2)}")  # slightly better
print(f"Accuracy model2: {accuracy_score(model2.predict(x_test2), y_test2)}")

# Accuracy model1: 0.8208333333333333
# Accuracy model2: 0.8205

# The best model is
# AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3), n_estimators=80, learning_rate=0.1)
