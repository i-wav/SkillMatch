import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

df_tech2 = pd.read_csv("./data/df_tech2.csv")

vectorizer = joblib.load("./models/vectorizer.pkl")
tfidf_matrix = joblib.load("./models/tfidf_matrix.pkl")

def recommend_job_titles(user_skills):
    user_skills_tfidf = vectorizer.transform([user_skills])
    cosine_sim = cosine_similarity(user_skills_tfidf, tfidf_matrix)

    top_indices = cosine_sim.argsort()[0][-10:][::-1]  
    recommendations = df_tech2.iloc[top_indices]["CLEANED_JOB_TITLE"].unique()[:3]

    return recommendations

# Streamlit UI
st.title("Job Recommendation System")
user_input = st.text_area("Enter your skills:", "")

if st.button("Get Recommendations"):
    if user_input:
        recommendations = recommend_job_titles(user_input)
        st.write("### Recommended Job Titles:")
        for job in recommendations:
            st.write(f"- {job}")
    else:
        st.warning("Please enter your skills.")
