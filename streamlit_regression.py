import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import pandas as pd

from tensorflow.keras.models import load_model
model=load_model('regression_model.h5')
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder=pickle.load(file)
with open('onehot_encoder_geo.pkl','rb') as file:
    ohe=pickle.load(file)
with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)


# Streamlit app
st.title("Estimated Salary Prediction")
geography = st.selectbox('Geography', ohe.categories_[0])
gender = st.selectbox('Gender', label_encoder.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance', min_value=0.0, format="%.2f")
credit_score = st.number_input('Credit Score', min_value=0)

tenure = st.slider('Tenure', 0, 10)
num_of_products = st.selectbox('Number of Products', [1, 2, 3, 4])
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
exited=st.selectbox('Exited',[0, 1])


input_data=pd.DataFrame({

    'CreditScore':[credit_score],
    'Gender':[label_encoder.transform([gender])[0]],
    
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'Exited':[exited]
}
    
)
##One-hot encode 'Geography'
geo_encoded=ohe.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=ohe.get_feature_names_out(['Geography']))
##Combine ohe columns with input data
input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)
##Scale the input data
input_scaled = scaler.transform(input_data)
#Predict churn
prediction=model.predict(input_scaled)[0][0]

st.write("The Predicted Estimated Salary",prediction)
