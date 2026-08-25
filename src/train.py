from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix
)
import joblib
from data_prep import (
    scaler,
)

def train_logistic_regression(x_train, y_train):
    model = Pipeline([
        ("scaler", scaler()),  
        ("logreg", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ))
    ])
    return train_model(model, x_train, y_train)

def train_knn(x_train, y_train):
    model = Pipeline([
        ("scaler", scaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5))
    ])
    return train_model(model, x_train, y_train)

def train_decision_tree(x_train, y_train):
    model = DecisionTreeClassifier(
        max_depth=5,
        class_weight="balanced",
        random_state=42
    )
    return train_model(model, x_train, y_train)

def train_neural_network_sklearn(x_train, y_train):
    model = Pipeline([
        ("scaler", scaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            activation='relu',
            max_iter=200,
            random_state=42,
            early_stopping=True
        ))
    ])
    return train_model(model, x_train, y_train)

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }
    return results

def cross_validate_model(model, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["precision", "recall", "f1"]
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
    results = {
        "mean_precision": scores["test_precision"].mean(),
        "mean_recall": scores["test_recall"].mean(),
        "mean_f1": scores["test_f1"].mean()
    }
    return results

def compare_hyperparameters_KNN(X_train, X_test, y_train, y_test):
    results = []
    for k in [1, 5, 20]:
        model = Pipeline([
            ("scaler", scaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k))
        ])
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results.append({
            "k": k,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"]
        })
    return results
    
def compare_thresholds(model, X_test, y_test):
    probabilities = model.predict_proba(X_test)[:, 1]
    results = []
    for threshold in [0.3, 0.5, 0.7]:
        y_pred = (probabilities >= threshold).astype(int)
        results.append({
            "threshold": threshold,
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0)
        })
    return results

def evaluate_neural_network(model, X_test, y_test):
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob >= 0.5).astype(int)
    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }
    return results


def cross_validate_neural_network(X, y, n_splits=5):


    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    model = Pipeline([
        ("scaler", scaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            activation='relu',
            max_iter=200,
            random_state=42,
            early_stopping=True,
            verbose=False
        ))
    ])
    
    # Cross Validation
    scoring = ["precision", "recall", "f1"]
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
    
    results = {
        "mean_precision": np.mean(scores["test_precision"]),
        "mean_recall": np.mean(scores["test_recall"]),
        "mean_f1": np.mean(scores["test_f1"])
    }
    return results

def save_model(model, path):
    joblib.dump(model, path)