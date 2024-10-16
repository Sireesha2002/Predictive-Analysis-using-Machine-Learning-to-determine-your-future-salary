import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map


def clean_experience(x):
    if x ==  'More than 50 years':
        return 50
    if x == 'Less than 1 year':
        return 0.5
    return float(x)


def clean_education(x):
    if 'Bachelor’s degree' in x:
        return 'Bachelor’s degree'
    if 'Master’s degree' in x:
        return 'Master’s degree'
    if 'Professional degree' in x or 'Other doctoral' in x:
        return 'Post grad'
    return 'Less than a Bachelors'

def clean_Undergrad(x):
    if 'Computer science' in x:
        return 'Computer science'
    if 'Mathematics' in x:
        return 'Mathematics'
    if 'information technology' in x:
        return 'information technology'
    if 'Another engineering discipline' in x:
        return 'Another engineering discipline'
    if 'A business discipline' in x:
        return 'A business discipline'
    if 'I never declared a major' in x:
        return 'Id rather not say'
    if 'Web development' in x:
        return 'Web development'
    if 'A natural science' in x or 'Fine arts or performing arts' in x or 'A humanities discipline' in x or 'A social science' in x or'A health science' in x:
        return 'others'




@st.cache
def load_data():
    df = pd.read_csv("survey_results_public.csv")
    df = df[["Country", "EdLevel", "YearsCodePro", "Employment", "ConvertedComp","UndergradMajor"]]
    df = df[df["ConvertedComp"].notnull()]
    df = df.dropna()
    df = df[df["Employment"] == "Employed full-time"]
    df = df.drop("Employment", axis=1)

    country_map = shorten_categories(df.Country.value_counts(), 400)
    df["Country"] = df["Country"].map(country_map)
    df = df[df["ConvertedComp"] <= 250000]
    df = df[df["ConvertedComp"] >= 10000]
    df = df[df["Country"] != "Other"]

    df["YearsCodePro"] = df["YearsCodePro"].apply(clean_experience)
    df["EdLevel"] = df["EdLevel"].apply(clean_education)
    df['UndergradMajor'] = df['UndergradMajor'].apply(clean_Undergrad)
    df = df.rename({"ConvertedComp": "Salary"}, axis=1)

    return df

df = load_data()

def show_explore_page():
    st.title("Data Analysis based on the features")

    st.write(
        """
    ### Data taken from Stack Overflow Developer Survey 2020
    """
    )

    data = df["Country"].value_counts()

    fig1, ax1 = plt.subplots()
    ax1.pie(data, labels=data.index, autopct="%1.1f%%", shadow=False, startangle=90)
    ax1.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.write("""#### Number of Data from different countries""")

    st.pyplot(fig1)
    
    st.write(
        """
    #### Mean Salary Based On Country
    """
    )

    data = df.groupby(["Country"])["Salary"].mean().sort_values(ascending=True)
    st.bar_chart(data)

    st.write(
        """
    #### Mean Salary Based On Experience
    """
    )

    data = df.groupby(["YearsCodePro"])["Salary"].mean().sort_values(ascending=True)
    st.line_chart(data)

    st.write(
        """
        #### Mean Salary Based on Major
    """
    )
    data = df.groupby(["UndergradMajor"])["Salary"].mean().sort_values(ascending=True)
    st.line_chart(data)


