import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import pandas as pd


model=tf.keras.models.load_model('model.keras')

##load the encoder and scalar
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

#load tje one hot encoder pickle file
with open('one_hot_encoder_.pkl','rb') as file:
    one_hot_enco=pickle.load(file)

#load the standadization pickle file
with open('scalar.pkl','rb') as file:
    scalar=pickle.load(file)

st.title('customer churn prediction')
geography=st.selectbox('geography',one_hot_enco.categories_[0])
gender=st.selectbox('gender',label_encoder_gender.classes_)
age=st.slider('Age',18,89)
balance=st.number_input('balance')
credict_score=st.number_input('credict_score')
estimated_salary=st.number_input('estimated_salary')
tenure=st.slider('tenure',0,10)
num_of_products=st.slider('numofproducts',1,4)
has_cr_card=st.selectbox('has_cr_card',[0,1])
is_active_member=st.selectbox('is_active _num,ber',[0,1])

if st.button('predict'):
    input_data=pd.DataFrame(
        {
            'CreditScore':[credict_score],
            'Gender':[label_encoder_gender.transform([gender])[0]],
            'Age':[age],
            'Tenure':[tenure],
            'Balance':[balance],
            'NumOfProducts':[num_of_products],
            'HasCrCard':[has_cr_card],
            'IsActiveMember':[is_active_member],
            'EstimatedSalary':[estimated_salary]
        }
    )
    geo_encoder=one_hot_enco.transform([[geography]]).toarray()
    geo_encoded_df=pd.DataFrame(geo_encoder,columns=one_hot_enco.get_feature_names_out(['Geography']))
    input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)
    input_data_scaled=scalar.transform(input_data)
    prediction=model.predict(input_data_scaled)
    prediction_proba=prediction[0][0]
    st.write(f'prediction probability of churn:{prediction_proba:.2f}')
    if prediction_proba>0.5:
        st.write("the customer is likely to churn")
    else:
        st.write("the customer is not likely to churn")