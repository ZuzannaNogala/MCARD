import argparse
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


def main():
    parser = argparse.ArgumentParser(description="Train LogisticRegression on processed "
                                                 "classification task and save predictions.")
    parser.add_argument('--x_train', type=str, default="data_train_x_task.csv",
                        help="Path to train features CSV file. (default: %(default)s)" )
    parser.add_argument('--y_train', type=str, default="data_train_y_task.csv",
                        help="Path to train labels CSV file. (default: %(default)s)")
    parser.add_argument('--x_test', type=str, default="data_test_x_task.csv",
                        help="Path to test features CSV file. (default: %(default)s)")
    parser.add_argument('--predictions', type=str, default="student331738_pred.csv",
                        help="Path to save predictions as CSV. (default: %(default)s)")
    args = parser.parse_args()

    # Load data
    X_train = pd.read_csv(args.x_train)
    y_train = pd.read_csv(args.y_train)["class"]

    X_test = pd.read_csv(args.x_test)

    # Fit AdaBoost - the best model version
    model = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3), n_estimators=80, learning_rate=0.1)
    model.fit(X_train, y_train)

    # Make predictions (predict class label)
    y_pred = model.predict(X_test)

    # Save predictions to CSV
    pd.Series(y_pred, name="class").to_csv(args.predictions, index=False, header=True)
    print(f"Predictions saved to {args.predictions}")


if __name__ == "__main__":
    main()
