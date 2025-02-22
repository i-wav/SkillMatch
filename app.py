import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Load Data
df_tech2 = pd.read_csv("./data/df_tech2.csv")  # Ensure correct path
vectorizer = joblib.load("./models/vectorizer.pkl")
tfidf_matrix = joblib.load("./models/tfidf_matrix.pkl")

# Streamlit Page Config
st.set_page_config(page_title="Job Recommendation System", layout="wide")

# Function to Recommend Jobs
def recommend_job_titles(user_skills):
    user_skills_tfidf = vectorizer.transform([user_skills])
    cosine_sim = cosine_similarity(user_skills_tfidf, tfidf_matrix)

    top_indices = cosine_sim.argsort()[0][-10:][::-1]  
    recommended_jobs = df_tech2.iloc[top_indices][["CLEANED_JOB_TITLE", "job_skills", "job_link"]].drop_duplicates()

    return recommended_jobs["CLEANED_JOB_TITLE"].unique()[:3], recommended_jobs["job_skills"].dropna().tolist()

# Function to Suggest Additional Skills
def suggest_additional_skills(user_skills, job_skills_list):
    user_skills_set = set(user_skills.lower().split(", "))

    all_required_skills = set()
    for job_skills in job_skills_list:
        all_required_skills.update(job_skills.lower().split(", "))

    missing_skills = list(all_required_skills - user_skills_set)

    # Format skills to start with capital letters
    formatted_skills = [" ".join([word.capitalize() for word in skill.split()]) for skill in missing_skills]

    return formatted_skills[:10]  # Limit to 10 additional skills

# --- UI Design ---
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>🔍 Skill Match</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: #666;'>Find the best job roles based on your skills</h5>", unsafe_allow_html=True)
st.write("---")

# Create two sections
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("💡 Enter Your Skills")
    user_input = st.text_area("Skills (comma-separated):", placeholder="e.g., Python, Data Analysis, Machine Learning")
    
    if st.button("Get Recommendations", use_container_width=True):
        if user_input:
            recommended_jobs, job_skills_list = recommend_job_titles(user_input)
            missing_skills = suggest_additional_skills(user_input, job_skills_list)

            with col2:
                st.subheader("🎯 Recommended Job Titles")
                for job_title in recommended_jobs:
                    st.markdown(f"✅ **{job_title}**", unsafe_allow_html=True)

                if missing_skills:
                    st.subheader("📌 Additional Skills You Might Need")
                    for skill in missing_skills:
                        st.warning(f"🔹 {skill}")

        else:
            st.error("⚠️ Please enter your skills to get recommendations.")
