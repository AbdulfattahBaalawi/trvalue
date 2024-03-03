import streamlit as st

import numpy as np
import pandas as pd
import joblib

model = joblib.load('model_xgb.joblib')
st.title('Predict the Value of TR')
# PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
day = st.text_input("day", '1') 
month = st.selectbox("Choose Month", [1,2,3,4,5,6,7,8,9,10,11,12])
year  = st.text_input("year", '2023')

def predict(): 
    x = np.array([day,month,year]) 
    x=np.asarray(x).reshape(1, -1)
    y_pred_RF = model.predict(x)
    if y_pred_RF : 
        st.success('The estimated Value of 1$ is',y_pred_RF)

trigger = st.button('Predict', on_click=predict)