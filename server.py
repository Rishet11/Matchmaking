from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os

app = Flask(__name__)

# Get the absolute path to the CSV file
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "Freelancer_Profiles.csv")

# Load freelancer data
try:
    freelancer_df = pd.read_csv(csv_path)
    freelancer_df["combined_text"] = freelancer_df["skills"] + ", " + freelancer_df["style"] + ", " + freelancer_df["experience"]
except Exception as e:
    print(f"Error loading data: {e}")
    freelancer_df = None

# Define pools
skills_pool = [
    "Premiere Pro", "Final Cut Pro", "After Effects", "Color Grading",
    "Transitions", "VFX", "Motion Graphics", "Slow Motion", "Text Animation"
]
styles_pool = ["Cinematic", "Corporate", "Fast-Cut", "Viral", "Vlog"]
experience_levels = ["Beginner", "Intermediate", "Expert"]

def extract_keywords(text, skills_pool, styles_pool):
    text = text.lower()
    found_skills = [s for s in skills_pool if re.search(r'\b' + re.escape(s.lower()) + r'\b', text)]
    found_styles = [s for s in styles_pool if re.search(r'\b' + re.escape(s.lower()) + r'\b', text)]
    found_experience = [e for e in experience_levels if re.search(r'\b' + re.escape(e.lower()) + r'\b', text)]
    return found_skills, found_styles, found_experience

def get_top_freelancers(user_input, required_experience=None, top_n=10):
    if freelancer_df is None:
        return {"error": "Database is currently unavailable. Please try again later."}
    
    if not user_input:
        return {"error": "Please provide project requirements"}
    
    skills, styles, experience = extract_keywords(user_input, skills_pool, styles_pool)
    if not any([skills, styles, experience]):
        return {"error": "Please include some skills, styles, or experience level requirements"}

    query = ", ".join(skills + styles + (experience if experience else []))
    
    # Filter by experience if specified
    working_df = freelancer_df.copy()
    if required_experience:
        working_df = working_df[working_df['experience'] == required_experience]
        if working_df.empty:
            return {"error": f"No freelancers found with {required_experience} experience level"}

    vectorizer = TfidfVectorizer()
    freelancer_vectors = vectorizer.fit_transform(working_df["combined_text"])
    query_vector = vectorizer.transform([query])

    similarity_scores = cosine_similarity(query_vector, freelancer_vectors).flatten()
    top_indices = similarity_scores.argsort()[::-1][:top_n]

    # Use the correct column name 'name' from the CSV
    top_matches = working_df.iloc[top_indices][["name", "skills", "style", "experience"]].copy()
    top_matches["similarity_score"] = (similarity_scores[top_indices] * 100).round(1)
    
    # Filter out low relevance matches
    top_matches = top_matches[top_matches["similarity_score"] > 20]
    
    if top_matches.empty:
        return {"error": "No suitable matches found"}
        
    return top_matches.reset_index(drop=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if freelancer_df is None:
        return render_template("index.html", 
                            error="Database is currently unavailable. Please try again later.",
                            experience_levels=experience_levels,
                            top_freelancers=None)

    if request.method == "POST":
        user_input = request.form.get("user_input", "").strip()
        experience_filter = request.form.get("experience")
        
        if not user_input:
            return render_template("index.html", 
                                error="Please provide project requirements", 
                                experience_levels=experience_levels,
                                top_freelancers=None)
            
        top_freelancers = get_top_freelancers(user_input, experience_filter)
        
        if isinstance(top_freelancers, dict) and "error" in top_freelancers:
            return render_template("index.html",
                                error=top_freelancers["error"], 
                                user_input=user_input,
                                experience_levels=experience_levels,
                                top_freelancers=None)
            
        return render_template("index.html",
                             top_freelancers=top_freelancers, 
                             user_input=user_input,
                             experience_levels=experience_levels)
    
    return render_template("index.html",
                         experience_levels=experience_levels,
                         top_freelancers=None)

if __name__ == "__main__":
    if freelancer_df is None:
        print("Warning: Could not load freelancer database. Application may not function correctly.")
    app.run(debug=True)