import streamlit as st
import joblib
import numpy as np

scaler = joblib.load('scaler.pkl')
model = joblib.load('model.pkl')

st.title('House Price Prediction')

st.divider()

bed = st.number_input('Bedrooms', value=2 , step=1)
bath = st.number_input('Bathrooms', value=1, step=1)
house_size = st.number_input('House Size', value=1000, step=50)

X = [bed, bath,house_size]

st.divider()

predict_btn = st.button('Predict')
st.divider()

if predict_btn:
    st.balloons()   
    X1= np.array(X)
    X_array = scaler.transform([X1])
    prediction = model.predict(X_array)[0]
    st.write(f'Predicted Price: {prediction:.2f}')
else:
    st.write('Click the button to predict the price')
    
    


