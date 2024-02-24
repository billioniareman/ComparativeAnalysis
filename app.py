# main.py
import streamlit as st
import os
import pandas as pd
from model import train_test_split_data, train_random_forest, train_logistic_regression, train_decision_tree, \
    train_naive_bayes
import home
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Rain Data Hub",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)
def fetch_feature(df):
    return df.columns


def main():
    menu = ["Home", "Compare Algos", "Visualize Algos"]
    choice = st.sidebar.selectbox("Go to", menu)

    if choice == "Home":
        home.main()

    elif choice == "Compare Algos":
        st.subheader("Compare Algorithms Page")

        # Fetch uploaded dataset file paths from session state
        uploaded_files = st.session_state.get('uploaded_files', [])

        # Select dataset from the uploaded datasets
        selected_dataset = st.selectbox("Select a dataset:", uploaded_files)

        if selected_dataset:
            df = pd.read_csv(selected_dataset)
            feature = fetch_feature(df)
            selected_feature = st.multiselect("Choose a Feature", feature, default=None)
            target_options = st.selectbox("Choose a target", ['rainfall'])
            X = df[selected_feature]
            y = df[target_options]
            X_train, X_test, y_train, y_test = train_test_split_data(df=X, target_column=y)
            st.subheader("Performance Measures of Machine Learning Models:")
            st.write("Random Forest:")
            rf_metrics = train_random_forest(X_train, X_test, y_train, y_test)
            st.write(rf_metrics)

            st.write("Logistic Regression:")
            lr_metrics = train_logistic_regression(X_train, X_test, y_train, y_test)
            st.write(lr_metrics)

            st.write("Decision Tree:")
            dt_metrics = train_decision_tree(X_train, X_test, y_train, y_test)
            st.write(dt_metrics)

            st.write("Naive Bayes:")
            nb_metrics = train_naive_bayes(X_train, X_test, y_train, y_test)
            st.write(nb_metrics)
        else:
            st.warning("Please upload a dataset on the Home page.")

    elif choice == "Visualize Algos":
        st.subheader("Visualize Algorithms Page")

        # Fetch uploaded dataset file paths from session state
        uploaded_files = st.session_state.get('uploaded_files', [])

        # Select dataset from the uploaded datasets
        selected_dataset = st.selectbox("Select a dataset:", uploaded_files)

        if selected_dataset:
            df = pd.read_csv(selected_dataset)
            feature = fetch_feature(df)
            selected_feature = st.multiselect("Choose a Feature", feature, default=None)
            target_options = st.selectbox("Choose a target", ['rainfall'])
            X = df[selected_feature]
            y = df[target_options]
            X_train, X_test, y_train, y_test = train_test_split_data(df=X, target_column=y)

            # Calculate performance metrics for each algorithm
            rf_metrics = train_random_forest(X_train, X_test, y_train, y_test)
            lr_metrics = train_logistic_regression(X_train, X_test, y_train, y_test)
            dt_metrics = train_decision_tree(X_train, X_test, y_train, y_test)
            nb_metrics = train_naive_bayes(X_train, X_test, y_train, y_test)

            # Collect performance metrics for plotting
            algorithms = ['Random Forest', 'Logistic Regression', 'Decision Tree', 'Naive Bayes']
            accuracy_scores = [rf_metrics['accuracy'], lr_metrics['accuracy'], dt_metrics['accuracy'],
                               nb_metrics['accuracy']]
            precision_scores = [rf_metrics['precision'], lr_metrics['precision'], dt_metrics['precision'],
                                nb_metrics['precision']]
            recall_scores = [rf_metrics['recall'], lr_metrics['recall'], dt_metrics['recall'], nb_metrics['recall']]
            f1_scores = [rf_metrics['f1score'], lr_metrics['f1score'], dt_metrics['f1score'], nb_metrics['f1score']]

            # Plot performance comparison
            plot_performance_comparison(algorithms, accuracy_scores, precision_scores, recall_scores, f1_scores)


def plot_performance_comparison(algorithms, accuracy_scores, precision_scores, recall_scores, f1_scores):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    ax[0, 0].bar(algorithms, accuracy_scores, color='skyblue')
    ax[0, 0].set_title('Accuracy')
    ax[0, 1].bar(algorithms, precision_scores, color='orange')
    ax[0, 1].set_title('Precision')
    ax[1, 0].bar(algorithms, recall_scores, color='green')
    ax[1, 0].set_title('Recall')
    ax[1, 1].bar(algorithms, f1_scores, color='red')
    ax[1, 1].set_title('F1 Score')

    plt.tight_layout()
    st.pyplot(fig)


if __name__ == "__main__":
    main()
