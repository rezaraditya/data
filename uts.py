import pickle
import streamlit as st

model = pickle.load(open('diabet.sav', 'rb'))

st.title ('Info diabetes')

Glucose = st.number_input('input glukosa ')
BloodPressure = st.number_input('input glukosa ')
SkinThickness = st.number_input('input skin ')	
Insulin	= st.number_input('input insul ')
BMI	= st.number_input('input glukosa ')
DiabetesPedigreeFunction = st.number_input('input dpf ')
Age	= st.number_input('input umur ')
Intercept= st.number_input('input intersep ')

predict = ''


if st.button('cek') :
    predict = model.predict(
        [[Glucose,BloodPressure,SkinThickness,Insulin,BMI, DiabetesPedigreeFunction, Age, Intercept]]
    )
    st.write('hasil : ', predict)
