import streamlit as st
import requests
from PIL import Image


st.title("Employee Churn Prediction")
st.subheader("Just enter variabel below then click Predict button :sunglasses:")

with st.form(key = "air_data_form"):
    
    Education = st.selectbox(
        label = "1. Pendidikan Terakhir: ",
        options = (
            "Bachelors",
            "Masters",
            "PHD"
        )
    )

    City = st.selectbox(
        label = "2. Kota Asal: ",
        options = (
            "Bangalore",
            "Pune",
            "New Delhi"
        )
    )

    Gender = st.radio(
        label = "3. Jenis Kelamin: ",
        options = (
            "Male",
            "Female"
        )
    )

    EverBenched = st.radio(
        label = "4. Apakah Anda Pernah Diberhentikan?",
        options = (
            "Yes",
            "No",
        )
    )

    JoiningYear = st.number_input(
        label = "5. Tahun Bergabung: ",
        min_value = 2012,
        max_value = 2018,
        help = "Value range from 2012 to 2018"
    )

    PaymentTier = st.number_input(
        label = "6. Tingkat Pembayaran Anda: ",
        min_value = 1,
        max_value = 3,
        help = "Value range from 1 to 3"
    )

    Age = st.slider(
        label = "7. Usia: ",
        min_value = 22,
        max_value = 41,
        help = "Value range from 22 to 41"
    )

    ExperienceInCurrentDomain = st.slider(
        label = "8. Pengalaman Anda pada Domain ini (dalam tahun): ",
        min_value = 0,
        max_value = 7,
        help = "Value range from 0 to 7"
    )
    
    submitted = st.form_submit_button("Predict")

    if submitted:
        raw_data = {
            "Education": Education,
            "City": City,
            "Gender": Gender,
            "EverBenched": EverBenched,
            "JoiningYear": JoiningYear,
            "PaymentTier": PaymentTier,
            "Age": Age,
            "ExperienceInCurrentDomain": ExperienceInCurrentDomain
        }

        with st.spinner("Sending data to prediction server ..."):
            res = requests.post("http://localhost:8080/predict", json = raw_data).json()
            # res = requests.post("http://api:8080/predict", json = raw_data).json()

        if res["error_msg"] != "":
            st.error("Error Occurs While Predicting: {}".format(res["error_msg"]))
        else:
            if res["res"] != "Churn":
                st.warning("Predicted: Not Churn.")
            else:
                st.success("Predicted: Churn.")

