# AI Resume Screening System

## Overview

This project is an AI-powered resume screening system that automates candidate evaluation based on a given job description. It reduces manual effort and provides structured insights for faster hiring decisions.

## Features

* Upload multiple resumes (PDF)
* AI-based candidate evaluation (OpenAI)
* Hybrid scoring (AI + keyword matching)
* Candidate ranking system
* Strengths & gaps analysis
* Final recommendation (Strong / Moderate / Not Fit)
* Streamlit web interface
* Basic login authentication

## Tech Stack

* Python
* OpenAI API
* Streamlit
* Pandas
* PyPDF2

## How It Works

1. User inputs a job description
2. Uploads candidate resumes (PDF)
3. System extracts text from resumes
4. AI evaluates each resume against JD
5. Hybrid scoring is calculated
6. Results are displayed in a ranked table

## Installation

```bash
git clone https://github.com/your-username/ai-resume-screening.git
cd ai-resume-screening
pip install -r requirements.txt
```

## Run Locally

```bash
streamlit run app.py
```

## Environment Variables

Create a `.env` file:

```
OPENAI_API_KEY=your_api_key
```

## Output Example

| Candidate | Score | Strengths    | Gaps    | Recommendation |
| --------- | ----- | ------------ | ------- | -------------- |
| A         | 82    | Python, APIs | Weak ML | Strong Fit     |

## Future Improvements

* Email automation for shortlisted candidates
* Advanced NLP skill extraction
* ATS integration
* Admin dashboard

## Demo

(Add your deployed app link here)

## Author

Aayush Kumar Singh
