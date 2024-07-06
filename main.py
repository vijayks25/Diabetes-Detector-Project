import streamlit as st
import pandas as pd
import joblib
import sklearn

# Load the trained decision tree model
model = joblib.load('diabetes_model.pkl')

# Title of the app
st.title('Diabetes Prediction App')

# User inputs
pregnancies = st.number_input('Number of Pregnancies', min_value=0, value=0)
glucose = st.number_input('Glucose Level', min_value=0, value=120)
blood_pressure = st.number_input('Blood Pressure', min_value=0, value=70)
skin_thickness = st.number_input('Skin Thickness', min_value=0, value=20)
insulin = st.number_input('Insulin Level', min_value=0, value=85)
bmi = st.number_input('Body Mass Index', value=20.0)
dpf = st.number_input('Diabetes Pedigree Function', value=0.5)
age = st.number_input('Age', min_value=18, value=30)

# Button to make prediction
if st.button('Predict Diabetes'):
		# Create a DataFrame based on the inputs
		input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]],
															columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

		# Get the model's prediction
		prediction = model.predict(input_data)
		probability = model.predict_proba(input_data)[0][1]

		# Display the prediction
		if prediction[0] == 1:
				st.write(f'Based on the input, there is a {probability*100:.2f}% chance that you have diabetes.')
		else:
				st.write(f'Based on the input, there is a {probability*100:.2f}% chance that you do not have diabetes.')

# Display model accuracy (you can add the actual accuracy value if known)
st.write("Note: This prediction is based on a model trained with limited data. Please consult a healthcare professional for a conclusive diagnosis.")
