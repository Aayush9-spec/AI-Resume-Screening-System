import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from groq import Groq
from PyPDF2 import PdfReader
import json
import sqlite3
import hashlib

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)''')
    conn.commit()
    conn.close()

init_db()

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def add_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('INSERT INTO users (username, password) VALUES (?,?)', (username, password))
    conn.commit()
    conn.close()

def login_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
    data = c.fetchall()
    conn.close()
    return data

def check_password():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        st.markdown("<h1 style='text-align: center;'>🔐 AI Recruiter Portal</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Please Log In or Sign Up to access the ATS dashboard.</p>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.write("")
            with st.container(border=True):
                tab1, tab2 = st.tabs(["Login", "Sign Up"])
                
                with tab1:
                    username = st.text_input("👤 Username", key="login_user")
                    password = st.text_input("🔑 Password", type="password", key="login_pass")
                    
                    if st.button("Log In", use_container_width=True, type="primary"):
                        if username and password:
                            hashed_pswd = make_hashes(password)
                            result = login_user(username, hashed_pswd)
                            if result:
                                st.session_state.logged_in = True
                                st.session_state.current_user = username
                                st.rerun()
                            else:
                                st.error("Incorrect username or password")
                        else:
                            st.warning("Please enter both username and password")
                            
                with tab2:
                    new_user = st.text_input("👤 New Username", key="reg_user")
                    new_password = st.text_input("🔑 New Password", type="password", key="reg_pass")
                    
                    if st.button("Sign Up", use_container_width=True):
                        if new_user and new_password:
                            conn = sqlite3.connect('users.db')
                            c = conn.cursor()
                            c.execute('SELECT * FROM users WHERE username = ?', (new_user,))
                            if c.fetchone():
                                st.warning("Username already exists! Please try logging in.")
                            else:
                                add_user(new_user, make_hashes(new_password))
                                st.success("Account created successfully! You can now log in.")
                            conn.close()
                        else:
                            st.warning("Please enter username and password")
        return False
    else:
        st.sidebar.title(f"👤 Welcome, {st.session_state.get('current_user', 'User')}")
        if st.sidebar.button("Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()
        return True

if not check_password():
    st.stop()

st.title("AI Resume Screening System")

jd = st.text_area("Paste Job Description")

uploaded_files = st.file_uploader(
    "Upload Resumes (PDF)",
    type=["pdf"],
    accept_multiple_files=True
)

def extract_text(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

def extract_keywords(jd):
    return [word.lower() for word in jd.split() if len(word) > 3]

def keyword_score(jd, resume):
    jd_words = set(extract_keywords(jd))
    resume_words = set(resume.lower().split())

    match = jd_words.intersection(resume_words)
    if not jd_words: return 0
    return int((len(match) / len(jd_words)) * 100)

def missing_skills(jd, resume):
    jd_words = set(extract_keywords(jd))
    resume_words = set(resume.lower().split())

    missing = jd_words - resume_words
    return list(missing)[:5]

def evaluate(jd, resume):
    prompt = f"""
You are an AI recruiter.

STRICT RULE:
Return ONLY valid JSON.
No explanation.

Format:
{{
"score": number,
"strengths": ["", "", ""],
"gaps": ["", "", ""],
"recommendation": ""
}}

JD:
{jd}

Resume:
{resume}
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )

        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

        return json.loads(content)
    except Exception as e:
        return {"score": 0, "strengths": [], "gaps": [f"API Error: {str(e)}"], "recommendation": "Error"}

if st.button("Evaluate Candidates"):
    if not jd:
        st.warning("Please paste a job description first.")
    elif not uploaded_files:
        st.warning("Please upload at least one resume.")
    else:
        results = []

        for file in uploaded_files:
            resume_text = extract_text(file)
            result = evaluate(jd, resume_text)

            ai_score = result.get("score", 0)
            kw_score = keyword_score(jd, resume_text)
            final_score = int((0.7 * ai_score) + (0.3 * kw_score))
            
            if result.get("recommendation") == "Error":
                st.error(f"Error evaluating {file.name}: {result.get('gaps', ['Unknown error'])[0]}")
                final_score = kw_score  # At least give them keyword score

            results.append({
                "Candidate": file.name,
                "Score": final_score,
                "Strengths": ", ".join(result.get("strengths", [])),
                "Gaps": ", ".join(missing_skills(jd, resume_text)) if result.get("recommendation") != "Error" else ", ".join(result.get("gaps", [])),
                "Recommendation": result.get("recommendation", "")
            })

        df = pd.DataFrame(results).sort_values(by="Score", ascending=False).reset_index(drop=True)
        
        # Add clear ranking column at the front
        df.insert(0, "Rank", ["🏆 #" + str(i+1) if i == 0 else "#" + str(i+1) for i in range(len(df))])
        
        if not df.empty and df.iloc[0]["Score"] > 0:
            st.success(f"🎉 **Top Candidate:** {df.iloc[0]['Candidate']} (Score: {df.iloc[0]['Score']})")
        
        st.dataframe(df)
        st.bar_chart(df.set_index("Candidate")["Score"])
