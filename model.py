# model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train_test_split_data(df, target_column, test_size=0.2, random_state=42):
    X = df
    y = target_column
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred, pos_label='yes')  # Update pos_label here
    recall = recall_score(y_test, y_pred, pos_label='yes')  # Update pos_label here
    accuracy = accuracy_score(y_test, y_pred)
    f1score = f1_score(y_test, y_pred,pos_label='yes')

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1score": f1score}

def train_logistic_regression(X_train, X_test, y_train, y_test):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred, pos_label='yes')  # Update pos_label here
    recall = recall_score(y_test, y_pred, pos_label='yes')  # Update pos_label here
    accuracy = accuracy_score(y_test, y_pred)
    f1score = f1_score(y_test, y_pred, pos_label='yes')

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1score": f1score}


def train_decision_tree(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred, pos_label='yes')  # Update pos_label here
    recall = recall_score(y_test, y_pred, pos_label='yes')  # Update pos_label here
    accuracy = accuracy_score(y_test, y_pred)
    f1score = f1_score(y_test, y_pred, pos_label='yes')

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1score": f1score}


def train_naive_bayes(X_train, X_test, y_train, y_test):
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred, pos_label='yes')  # Update pos_label here
    recall = recall_score(y_test, y_pred, pos_label='yes')  # Update pos_label here
    accuracy = accuracy_score(y_test, y_pred)
    f1score = f1_score(y_test, y_pred, pos_label='yes')

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1score": f1score}

