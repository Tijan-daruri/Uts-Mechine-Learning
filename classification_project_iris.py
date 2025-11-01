
"""classification_project_iris.py
Generated analysis script for Iris classification project.
- Models: Logistic Regression, Decision Tree
- Evaluations: confusion matrix, accuracy, precision, recall, f1, ROC (one-vs-rest)
"""

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, classification_report, roc_curve, auc
)
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

df = pd.DataFrame(X, columns=feature_names)
df['target'] = y
df['target_name'] = df['target'].map(lambda i: target_names[i])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

models = {
    'LogisticRegression': LogisticRegression(max_iter=200, multi_class='ovr'),
    'DecisionTree': DecisionTreeClassifier(random_state=42)
}

for name, model in models.items():
    if name == 'LogisticRegression':
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        y_prob = model.predict_proba(X_test_s)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
    print("\n=== Model:", name, "===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=target_names))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:\n", cm)
    # ROC per class (one-vs-rest)
    y_test_bin = label_binarize(y_test, classes=np.arange(len(target_names)))
    for i in range(y_test_bin.shape[1]):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        print(f"Class {target_names[i]} AUC = {{roc_auc:.3f}}")
