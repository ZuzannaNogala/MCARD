from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.datasets import load_iris


X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Trees

clf_tree_1 = DecisionTreeClassifier(max_depth=1)
clf_tree_2 = DecisionTreeClassifier(max_depth=2)
clf_tree_3 = DecisionTreeClassifier(max_depth=3)

clf_tree_1.fit(X_train, y_train)
clf_tree_2.fit(X_train, y_train)
clf_tree_3.fit(X_train, y_train)

# RandomForest

clf_RandomForest = RandomForestClassifier(n_estimators=100, max_depth=2)
clf_RandomForest.fit(X_train, y_train)

# AdaBoost

clf_Ada_tree_1 = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1),  n_estimators=50)
clf_Ada_tree_2 = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2),  n_estimators=50)
clf_Ada_tree_3 = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3),  n_estimators=50)

clf_Ada_RandomForest = AdaBoostClassifier(estimator=clf_RandomForest, n_estimators=50)

clf_Ada_tree_1.fit(X_train, y_train)
clf_Ada_tree_2.fit(X_train, y_train)
clf_Ada_tree_3.fit(X_train, y_train)
clf_Ada_RandomForest.fit(X_train, y_train)


# Accuracy:

def compute_accuracy(classifier, x_test, y_test, name_of_classifier):
    y_prediction = classifier.predict(X_test)
    print(f"{name_of_classifier}, ACC = ", accuracy_score(y_prediction, y_test))


classifier_list = [clf_tree_1, clf_tree_2, clf_tree_3,
                   clf_RandomForest,
                   clf_Ada_tree_1, clf_Ada_tree_2, clf_Ada_tree_3,
                   clf_Ada_RandomForest]

classifier_labels = ["TREE (max_depth=1)", "TREE (max_depth=2)", "TREE (max_depth=3)",
                     "RANDOM FOREST", "ADABOOST on TREE (max_depth=1)",
                     "ADABOOST on TREE (max_depth=2)", "ADABOOST on TREE (max_depth=3)",
                     "ADABOOST on RANDOM FOREST"]

for i in range(len(classifier_list)):
    compute_accuracy(classifier_list[i], X_test, y_test, classifier_labels[i])


# y_tree_1_prediction = clf_tree_1.predict(X_test)
# y_tree_2_prediction = clf_tree_2.predict(X_test)
# y_tree_3_prediction = clf_tree_3.predict(X_test)
#
# y_RandomForest_prediction = clf_RandomForest.predict(X_test)
#
# y_Ada_tree_1_prediction = clf_Ada_tree_1.predict(X_test)
# y_Ada_tree_2_prediction = clf_Ada_tree_2.predict(X_test)
# y_Ada_tree_3_prediction = clf_Ada_tree_3.predict(X_test)
# y_Ada_RandomForest_prediction = clf_Ada_RandomForest.predict(X_test)





