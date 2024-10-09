import streamlit as st
import pickle
import numpy as np


def load_model():
    with open('saved_steps_2.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]
le_UndergradMajor = data["le_UndergradMajor"]

def show_predict_page():
    st.title("EMPLOYEE SALARY PREDICTION")

    st.write("""### We need some information to predict the salary""")


    countries = (
        "United States",
        "India",
        "United Kingdom",
        "Germany",
        "Canada",
        "Brazil",
        "France",
        "Spain",
        "Australia",
        "Netherlands",
        "Poland",
        "Italy",
        "Russian Federation",
        "Sweden",
    )

    education = (
        "Less than a Bachelors",
        "Bachelor’s degree",
        "Master’s degree",
        "Post grad",
    )

    undergradmajor = (
        "Computer science",
        "Mathematics",
        "Another engineering discipline",
        "information technology",
        "Web development",
        "Id rather not say",
        "A business discipline",
    )

    country = st.selectbox("Country", countries)
    education = st.selectbox("Education Level", education)
    undergradmajor = st.selectbox("Major",undergradmajor)
    expericence = st.slider("Experience", 0, 50, 3)

    ok = st.button("Calculate Salary")
    if ok:
        X = np.array([[country, education, expericence, undergradmajor]])
        X[:, 0] = le_country.transform(X[:, 0])
        X[:, 1] = le_education.transform(X[:, 1])
        X[:, 3] = le_UndergradMajor.transform(X[:, 3])
        X = X.astype(float)
        X

        salary = regressor.predict(X)
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")