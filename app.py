
import streamlit as st
import pandas as pd
import pickle
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.utils import class_weight

# Load the trained model
loaded_model = pickle.load(open("C:/Users/aweso/OneDrive/Desktop/majorporject/trained_model.sav", 'rb'))

def predict_autism(age, jaundice, austim, result):
    # Custom prediction logic based on jaundice and austim
    if jaundice == 'Yes' and austim == 'Yes':
        return "The Patient Has Autism."
    elif jaundice == 'Yes' and austim == 'No':
        return 'The Patient does not have Autism.'
    elif jaundice == 'No' and austim == 'Yes':
        return 'The Patient Has Autism.'
    else:
        # Make prediction using the loaded model if none of the above conditions are met
        input_data = pd.DataFrame({
            'age': [age],
            'jaundice': [1 if jaundice == 'Yes' else 0],
            'austim': [1 if austim == 'Yes' else 0],
            'result': [result]
        })
        prediction = loaded_model.predict(input_data)
        return prediction[0]

def main():
    # Streamlit UI
    st.title('Autism Detection')
    st.write('Enter the following information to predict autism:')
    
    # Input fields
    age = st.number_input('Age', min_value=0, max_value=120, value=25)
    jaundice = st.selectbox('Jaundice', ['Yes', 'No'])
    austim = st.selectbox('Austim', ['Yes', 'No'])
    result = st.number_input('Result', min_value=0, max_value=30, value=5)


    
    # Make prediction when 'Predict' button is clicked
    if st.button('Predict'):
        prediction = predict_autism(age, jaundice, austim, result)
        st.write(f'The predicted result is: {prediction}')

 # Info button
    if st.sidebar.button('ℹ️ Info'):
        st.sidebar.write("""
        ## About This Project
        This project was developed for the purpose of detecting autism using machine learning techniques.
        
        ## Developer
       - Karan Gupta 
       - Kavya Gupta 
        """)

if __name__ == '__main__':
    main()