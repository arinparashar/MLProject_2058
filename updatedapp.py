import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load data and preprocess
data_path = 'Crop_recommendation.csv'  # Ensure this is the correct path to your dataset
crop_data = pd.read_csv(data_path)

# Encode the target labels
label_encoder = LabelEncoder()
crop_data['label_encoded'] = label_encoder.fit_transform(crop_data['label'])
features = crop_data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
labels_encoded = crop_data['label_encoded']

# Initialize models
dt_classifier = DecisionTreeClassifier(criterion="entropy", random_state=2, max_depth=5)
rf_classifier = RandomForestClassifier(random_state=2, n_estimators=100)
log_reg = LogisticRegression(max_iter=200)
nb_classifier = GaussianNB()
xgb_classifier = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=2)

# Train models
dt_classifier.fit(features, labels_encoded)
rf_classifier.fit(features, labels_encoded)
log_reg.fit(features, labels_encoded)
nb_classifier.fit(features, labels_encoded)
xgb_classifier.fit(features, labels_encoded)

# Streamlit UI
st.title("AgriSmart – Precision Crop Advisor")

# Input sliders for feature values
N = st.slider("Nitrogen (N)", 0, 140, step=1)
P = st.slider("Phosphorus (P)", 0, 145, step=1)
K = st.slider("Potassium (K)", 0, 205, step=1)
temperature = st.slider("Temperature (°C)", 0.0, 50.0, step=0.1)
humidity = st.slider("Humidity (%)", 0.0, 100.0, step=0.1)
ph = st.slider("pH Level", 0.0, 14.0, step=0.1)
rainfall = st.slider("Rainfall (mm)", 0.0, 300.0, step=0.1)

# Prepare data for prediction
input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], 
                          columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])

# Prediction button
if st.button("Recommend Crop"):
    # Make predictions for each model
    dt_prediction_encoded = dt_classifier.predict(input_data)
    rf_prediction_encoded = rf_classifier.predict(input_data)
    log_reg_prediction_encoded = log_reg.predict(input_data)
    nb_prediction_encoded = nb_classifier.predict(input_data)
    xgb_prediction_encoded = xgb_classifier.predict(input_data)
    
    # Decode predictions to get crop names
    dt_prediction = label_encoder.inverse_transform(dt_prediction_encoded)[0]
    rf_prediction = label_encoder.inverse_transform(rf_prediction_encoded)[0]
    log_reg_prediction = label_encoder.inverse_transform(log_reg_prediction_encoded)[0]
    nb_prediction = label_encoder.inverse_transform(nb_prediction_encoded)[0]
    xgb_prediction = label_encoder.inverse_transform(xgb_prediction_encoded)[0]
    
    # Calculate accuracy for each model
    dt_accuracy = accuracy_score(labels_encoded, dt_classifier.predict(features)) * 100
    rf_accuracy = accuracy_score(labels_encoded, rf_classifier.predict(features)) * 100
    log_reg_accuracy = accuracy_score(labels_encoded, log_reg.predict(features)) * 100
    nb_accuracy = accuracy_score(labels_encoded, nb_classifier.predict(features)) * 100
    xgb_accuracy = accuracy_score(labels_encoded, xgb_classifier.predict(features)) * 100

    # Display recommendations and accuracy for each model
    st.subheader("Model Recommendations:")
    st.write(f"**Decision Tree** recommends: `{dt_prediction}` with accuracy of {dt_accuracy:.2f}%")
    st.write(f"**Random Forest** recommends: `{rf_prediction}` with accuracy of {rf_accuracy:.2f}%")
    st.write(f"**Logistic Regression** recommends: `{log_reg_prediction}` with accuracy of {log_reg_accuracy:.2f}%")
    st.write(f"**Naive Bayes** recommends: `{nb_prediction}` with accuracy of {nb_accuracy:.2f}%")
    st.write(f"**XGBoost** recommends: `{xgb_prediction}` with accuracy of {xgb_accuracy:.2f}%")
    
    # Plotting the accuracies of each model
    model_names = ['Decision Tree', 'Random Forest', 'Logistic Regression', 'Naive Bayes', 'XGBoost']
    accuracies = [dt_accuracy, rf_accuracy, log_reg_accuracy, nb_accuracy, xgb_accuracy]

    fig, ax = plt.subplots()
    ax.barh(model_names, accuracies, color=['#6A1B9A', '#1976D2', '#388E3C', '#D32F2F', '#FFA000'])
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Model Accuracy Comparison")
    for i, (acc, name) in enumerate(zip(accuracies, model_names)):
        ax.text(acc + 1, i, f"{acc:.2f}%", va='center')  # Adding accuracy values next to the bars

    st.pyplot(fig)
