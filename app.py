import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(layout="wide", page_title="ECS 171 Team Project - Group 5")

df = pd.read_csv('heart.csv')

st.title('ECS 171 Team Project - Group 5')
st.write('In this project, we will be analyzing the heart disease dataset. This dataset is from Kaggle. Here is a preliminary look at the dataset:')
st.dataframe(df)

# Prepare the data
heart_df_encoded = pd.get_dummies(df, columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'])
X = heart_df_encoded.drop('HeartDisease', axis=1)
y = heart_df_encoded['HeartDisease']

# Initialize the scaler
scaler = StandardScaler()

# Define the model variable outside of the button click event
model = None

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
    if model_option == 'Neural Network':
        # Use the scaled data for Neural Network
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Initialize the model based on user selection
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

# Check if model is selected and trained before allowing prediction
if model:
    # Personalized Heart Disease Prediction
    st.title('Personalized Heart Disease Prediction')
    st.write('Please enter your data:')

    age = st.number_input('Age', min_value=0, max_value=120, value=30, step=1)
    sex = st.selectbox('Sex', ['M', 'F'])
    chest_pain_type = st.selectbox('Chest Pain Type', ['ATA', 'NAP', 'ASY', 'TA'])
    resting_bp = st.number_input('Resting Blood Pressure', min_value=0, max_value=300, value=120, step=1)
    cholesterol = st.number_input('Cholesterol', min_value=0, max_value=1000, value=200, step=1)
    fasting_bs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1])
    resting_ecg = st.selectbox('Resting Electrocardiographic Results', ['Normal', 'ST', 'LVH'])
    max_hr = st.number_input('Maximum Heart Rate Achieved', min_value=0, max_value=220, value=100, step=1)
    exercise_angina = st.selectbox('Exercise-induced Angina', ['Y', 'N'])
    oldpeak = st.number_input('Oldpeak', value=0.0, step=0.1)
    st_slope = st.selectbox('ST Slope', ['Up', 'Flat', 'Down'])

    user_data = {
        'Age': [age],
        'Sex': [sex],
        'ChestPainType': [chest_pain_type],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs],
        'RestingECG': [resting_ecg],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exercise_angina],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope]
    }

    user_df = pd.DataFrame.from_dict(user_data)
    user_df_encoded = pd.get_dummies(user_df, columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'])

    # Ensure all columns are present
    for col in X.columns:
        if col not in user_df_encoded.columns:
            user_df_encoded[col] = 0

    user_df_encoded = user_df_encoded.reindex(columns=X.columns)

    if model_option == 'Neural Network':
        user_df_encoded_scaled = scaler.transform(user_df_encoded)
        prediction = model.predict(user_df_encoded_scaled)
    else:
        prediction = model.predict(user_df_encoded)

    if prediction[0] == 0:
        st.write('Based on the input data, the model predicts: No Heart Disease')
    else:
        st.write('Based on the input data, the model predicts: Heart Disease')
