import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings("ignore")

# Load datasets
def load_data(train_path, test_path):
    train_dataset = pd.read_csv(train_path)
    test_dataset = pd.read_csv(test_path)
    X_train = train_dataset.iloc[:, 2:19].values
    y_train = train_dataset.iloc[:, 19].values
    X_test = test_dataset.iloc[:, 2:19].values
    y_test = test_dataset.iloc[:, 19].values
    return X_train, y_train, X_test, y_test

# Transform data function
def transform_data(X, y, X_new, mul_vals):
    X_class_0 = X[y == 0]
    X_class_1 = X[y == 1]

    freq_0 = np.zeros((X.shape[1], 26))
    freq_1 = np.zeros((X.shape[1], 26))
    
    for i in range(X_class_0.shape[0]):
        for j in range(X_class_0.shape[1]):
            idx = ord(X_class_0[i, j]) - ord('A')
            freq_0[j, idx] += 1
    
    for i in range(X_class_1.shape[0]):
        for j in range(X_class_1.shape[1]):
            idx = ord(X_class_1[i, j]) - ord('A')
            freq_1[j, idx] += 1
    
    freq_ratio = (freq_0 + 0.001) / (freq_1 + 0.001)

    X_transformed = np.zeros(X.shape)
    X_new_transformed = np.zeros(X_new.shape)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            idx = ord(X[i, j]) - ord('A')
            X_transformed[i, j] = np.log10(freq_ratio[j, idx] + 1)
    
    for i in range(X_new.shape[0]):
        for j in range(X_new.shape[1]):
            idx = ord(X_new[i, j]) - ord('A')
            X_new_transformed[i, j] = np.log10(freq_ratio[j, idx] + 1)

    X_transformed *= mul_vals
    X_new_transformed *= mul_vals

    return X_transformed, X_new_transformed

# Main function to train models and plot ROC curves
def main(train_path, test_path, output_img='roc_curve.png', mul_vals=1):
    X_train, y_train, X_test, y_test = load_data(train_path, test_path)
    X_train_transformed, X_test_transformed = transform_data(X_train, y_train, X_test, mul_vals)

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

    classifiers = {
        'Linear SVC': LinearSVC(C=0.1, class_weight='balanced', dual=False, loss='squared_hinge', max_iter=800000, penalty='l2', tol=1e-06),
        'Logistic Regression': LogisticRegression(class_weight=None, max_iter=160000, penalty='none', solver='lbfgs'),
        'Random Forest Classifier': RandomForestClassifier(bootstrap=False, class_weight=None, max_depth=10, max_features='auto', min_samples_leaf=4, min_samples_split=5, n_estimators=800)
    }

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

    for clf_name, clf in classifiers.items():
        clf.fit(X_train_transformed, y_train)

        tpr_values = []
        auc_scores = []

        for train_idx, val_idx in cv.split(X_train_transformed, y_train):
            X_train_cv, X_val_cv = X_train_transformed[train_idx], X_train_transformed[val_idx]
            y_train_cv, y_val_cv = y_train[train_idx], y_train[val_idx]

            clf.fit(X_train_cv, y_train_cv)

            if hasattr(clf, "decision_function"):
                y_score = clf.decision_function(X_val_cv)
            else:
                y_score = clf.predict_proba(X_val_cv)[:, 1]

            fpr, tpr, _ = roc_curve(y_val_cv, y_score)
            tpr_values.append(np.interp(np.linspace(0, 1, 100), fpr, tpr))
            auc_scores.append(auc(fpr, tpr))

        if hasattr(clf, "decision_function"):
            y_score_test = clf.decision_function(X_test_transformed)
        else:
            y_score_test = clf.predict_proba(X_test_transformed)[:, 1]
        fpr_test, tpr_test, _ = roc_curve(y_test, y_score_test)
        roc_auc_test = auc(fpr_test, tpr_test)

        mean_tpr = np.mean(tpr_values, axis=0)
        mean_auc = np.mean(auc_scores)
        std_auc = np.std(auc_scores)
        ax.plot(fpr_test, tpr_test, label=f'{clf_name} (CV Mean AUC = {mean_auc:.2f} ± {std_auc:.2f}, Test AUC = {roc_auc_test:.2f})')
        ax.fill_between(np.linspace(0, 1, 100), mean_tpr - np.std(tpr_values, axis=0), mean_tpr + np.std(tpr_values, axis=0), alpha=0.2)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curves')
    ax.legend(loc="lower right")
    plt.savefig(output_img, dpi=300)
    plt.show()
    print(f'ROC curve saved as {output_img}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TCR/MHC prediction models.")
    parser.add_argument("train_path", type=str, help="Path to the training dataset (CSV file).")
    parser.add_argument("test_path", type=str, help="Path to the testing dataset (CSV file).")
    parser.add_argument("--output_img", type=str, default="roc_curve.png", help="Filename for the output ROC curve image.")
    parser.add_argument("--mul_vals", type=float, default=1, help="Multiplicative value for transformation.")
    
    args = parser.parse_args()
    main(args.train_path, args.test_path, args.output_img, args.mul_vals)

