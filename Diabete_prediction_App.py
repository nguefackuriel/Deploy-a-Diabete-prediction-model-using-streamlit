import numpy as np
import streamlit as st
import pickle
from GDA import GDA


# Load our mnodel
model_load = pickle.load(open('my_model_diabete.sav', 'rb'))

# Define a prediction function

def prediction_model(input_data):
    
    input_data_as_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_array.reshape(1,-1)

    prediction = model_load.predict(input_data_reshaped)

    if prediction[0] == 0:
        return "The patient doesn't have diabete"
    else:
        return 'The patient has diabete'


def main():

    # Give a title to the App
    st.title('Diabete prediction App')


    # Getting the Input from the user
    Pregnancies = st.text_input('Pregnancies value')
    Glucose = st.text_input('Glucose value')
    BloodPressure = st.text_input('Blood pressure value')
    SkinThickness = st.text_input('SkinThickness value')
    Insulin = st.text_input('Insulin value')
    BMI = st.text_input('BMI Value')
    DiabetesPedigreeFunction = st.text_input('WDiabetesPedigreeFunction value')
    Age = st.text_input('Age of the patient')

    # variable of prediction

    result = ''

    if st.button('Flower test result'):
        result = prediction_model([float(Pregnancies),float(Glucose),float(BloodPressure),float(SkinThickness),float(Insulin),float(BMI),float(DiabetesPedigreeFunction),float(Age)])

    # Display the result
    st.success(result)

if __name__ == '__main__':
    main()