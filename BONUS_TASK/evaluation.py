import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score


def main():
    parser = argparse.ArgumentParser(description="Evaluate predictions by comparing to ground truth labels.")
    parser.add_argument('--file1', type=str, default='data_test_y_orig.csv',
                        help='CSV file with ground truth labels (default: %(default)s)')
    parser.add_argument('--file2', type=str, default='student331738_pred.csv',
                        help='CSV file with predictions (default: %(default)s)')
    args = parser.parse_args()

    # Read files, always with header "class"
    y_true = pd.read_csv(args.file1)["class"]
    y_pred = pd.read_csv(args.file2)["class"]

    # Check matching lengths
    assert len(y_true) == len(y_pred), f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}"

    print("sum(y_pred) = ", np.sum(y_pred))
    print("sum(y_true) = ", np.sum(y_true))

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
