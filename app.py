import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


st.set_page_config(layout="wide", page_title="ECS 171 Team Project - Group 5")

df = pd.read_csv('heart.csv')

st.title('ECS 171 Team Project - Group 5')

st.write('In this project, we will be analyzing the heart disease dataset. This dataset is from Kaggle. Here is a preliminary look at the dataset:')

st.dataframe(df)

st.write('First we did some EDA to the dataset, specifically the creation of a heatmap and attribute distribution plot. This allowed us to get a cleaner look at the data.')

st.write('Here is the heatmap:')

df_encoded = pd.get_dummies(df, columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'])

hm = sns.heatmap(df_encoded.corr(), annot=True, fmt='.2f', vmin=-1, vmax=1, center=0)

st.pyplot(hm.figure)

st.write('Here is the attribute distribtuion charts:')

fig, axes = plt.subplots(4, 3, figsize=(8, 8))
axes = axes.flatten()

for i, (col, data) in enumerate(df.items()):
    sns.histplot(data, kde=True, ax=axes[i])
    axes[i].set_title(col)

plt.tight_layout()

st.pyplot(fig)

scaler = StandardScaler()

heart_df_encoded = pd.get_dummies(df, columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'])
X = heart_df_encoded.drop('HeartDisease', axis=1)
y = heart_df_encoded['HeartDisease']

st.title('Machine Learning Model Performance Viewer')

model_option = st.selectbox('Select a model:', ['Logistic Regression', 'SVM', 'Naive Bayes', 'Decision Tree', 'Neural Network'])
test_size = st.slider('Test size ratio', 0.1, 0.9, 0.2)

if model_option == 'Logistic Regression':
    penalty = st.selectbox('Penalty', ['l2', 'none'])
elif model_option == 'SVM':
    kernel = st.selectbox('Kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
elif model_option == 'Decision Tree':
    criterion = st.selectbox('Criterion', ['gini', 'entropy'])
elif model_option == 'Neural Network':
    hidden_layer_sizes = st.text_input('Hidden layer sizes (e.g., 100 or 50,50)', '100')
    activation = st.selectbox('Activation function', ['logistic', 'relu', 'tanh'])

if st.button('Evaluate Model'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    if model_option == 'Logistic Regression':
        model = LogisticRegression(penalty=penalty, max_iter=10000, random_state=42)
    elif model_option == 'SVM':
        model = SVC(kernel=kernel)
    elif model_option == 'Naive Bayes':
        model = GaussianNB()
    elif model_option == 'Decision Tree':
        model = DecisionTreeClassifier(criterion=criterion, random_state=42)
    elif model_option == 'Neural Network':
        hidden_layers = tuple(map(int, hidden_layer_sizes.split(',')))
        model = MLPClassifier(hidden_layer_sizes=hidden_layers, activation=activation, max_iter=100000, random_state=42)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    st.write(f"Accuracy: {accuracy}")
    st.write(f"Precision: {precision}")
    st.write(f"Recall: {recall}")
    st.write(f"F1 Score: {f1}")
