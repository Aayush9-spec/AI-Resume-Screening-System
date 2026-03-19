"""
AI Resume Screening Pipeline
==============================
Evaluates resumes against a Job Description using OpenAI GPT-4o-mini.
Outputs ranked results to CSV, HTML dashboard, and terminal.

Usage:
  1. Set OPENAI_API_KEY in .env file
  2. Place JD in jd.txt
  3. Place resumes in resumes/ folder
  4. Run: python3 resume_screening.py
"""

import os
import json
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from PyPDF2 import PdfReader

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# ─── Load API Key ────────────────────────────────────────────────
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print()
    print("❌ ERROR: OPENAI_API_KEY not found!")
    print()
    print("Please set your API key:")
    print("  1. Create a .env file with: OPENAI_API_KEY=sk-...")
    print("  2. Or export it: export OPENAI_API_KEY='sk-...'")
    print()
    exit(1)

client = OpenAI(api_key=api_key)


# ─── Load Job Description ───────────────────────────────────────
with open("jd.txt", "r") as f:
    jd = f.read()


# ─── Evaluation Function ────────────────────────────────────────
def evaluate_resume(jd, resume_text, name):
    """Evaluate a single resume against the JD using GPT-4o-mini."""
    prompt = f"""
You are an expert AI recruiter.

Job Description:
{jd}

Resume:
{resume_text}

Evaluate this candidate and return STRICT JSON only (no markdown, no extra text):
{{
  "score": <number 0-100>,
  "strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
  "gaps": ["<gap 1>", "<gap 2>", "<gap 3>"],
  "recommendation": "<Strong Fit | Moderate Fit | Not Fit>"
}}

Scoring guide:
- 80-100: Strong Fit (matches most requirements)
- 50-79: Moderate Fit (some relevant skills but gaps)
- 0-49: Not Fit (lacks key requirements)
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    content = response.choices[0].message.content.strip()

    # Clean markdown code block wrappers if present
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {
            "score": 0,
            "strengths": ["Parsing error"],
            "gaps": ["Parsing error"],
            "recommendation": "Not Fit"
        }

    return {
        "Candidate": name,
        "Score": parsed["score"],
        "Strengths": ", ".join(parsed["strengths"]),
        "Gaps": ", ".join(parsed["gaps"]),
        "Recommendation": parsed["recommendation"]
    }


# ─── Generate HTML Dashboard ────────────────────────────────────
def generate_html_dashboard(df, output_path="output.html"):
    """Generate a beautiful HTML dashboard from the results DataFrame."""
    sorted_df = df.sort_values(by="Score", ascending=False).reset_index(drop=True)

    scores = sorted_df["Score"].tolist()
    avg_score = sum(scores) / len(scores) if scores else 0
    top_candidate = sorted_df.iloc[0]["Candidate"] if len(sorted_df) > 0 else "N/A"
    strong = len(sorted_df[sorted_df["Recommendation"] == "Strong Fit"])
    moderate = len(sorted_df[sorted_df["Recommendation"] == "Moderate Fit"])
    not_fit = len(sorted_df[sorted_df["Recommendation"] == "Not Fit"])

    def badge_color(rec):
        return {"Strong Fit": "#10b981", "Moderate Fit": "#f59e0b", "Not Fit": "#ef4444"}.get(rec, "#6b7280")

    def score_color(s):
        if s >= 80: return "#10b981"
        elif s >= 50: return "#f59e0b"
        else: return "#ef4444"

    rows = ""
    for i, row in sorted_df.iterrows():
        sc = score_color(row["Score"])
        bc = badge_color(row["Recommendation"])
        strengths = "".join(f'<span class="tag tg">{s.strip()}</span>' for s in row["Strengths"].split(","))
        gaps = "".join(f'<span class="tag tr">{g.strip()}</span>' for g in row["Gaps"].split(","))
        rows += f"""<tr>
            <td class="rk">#{i+1}</td>
            <td class="cn">{row["Candidate"]}</td>
            <td><div class="sc"><div class="sb"><div class="sf" style="width:{row['Score']}%;background:{sc}"></div></div><span style="color:{sc};font-weight:700">{row["Score"]}</span></div></td>
            <td><div class="tgs">{strengths}</div></td>
            <td><div class="tgs">{gaps}</div></td>
            <td><span class="bd" style="background:{bc}20;color:{bc};border:1px solid {bc}40">{row["Recommendation"]}</span></td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>AI Resume Screening Results</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Inter',sans-serif;background:#0a0a0f;color:#e2e8f0;min-height:100vh;padding:2rem}}
.ct{{max-width:1400px;margin:0 auto}}
.hd{{text-align:center;margin-bottom:3rem;padding:2rem 0}}
.hd h1{{font-size:2.5rem;font-weight:700;background:linear-gradient(135deg,#6366f1,#8b5cf6,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:.5rem}}
.hd p{{color:#94a3b8;font-size:1rem}}
.sts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1.5rem;margin-bottom:3rem}}
.st{{background:linear-gradient(135deg,rgba(99,102,241,.08),rgba(139,92,246,.05));border:1px solid rgba(99,102,241,.15);border-radius:16px;padding:1.5rem;backdrop-filter:blur(10px);transition:transform .2s}}
.st:hover{{transform:translateY(-2px);border-color:rgba(99,102,241,.3)}}
.sl{{font-size:.8rem;text-transform:uppercase;letter-spacing:.05em;color:#94a3b8;margin-bottom:.5rem}}
.sv{{font-size:2rem;font-weight:700}}
.sv.gr{{color:#10b981}}.sv.am{{color:#f59e0b}}.sv.rd{{color:#ef4444}}.sv.pr{{color:#a78bfa}}
.tc{{background:rgba(15,15,25,.6);border:1px solid rgba(99,102,241,.1);border-radius:20px;overflow:hidden;backdrop-filter:blur(10px)}}
.th{{display:flex;align-items:center;padding:1.5rem 2rem;border-bottom:1px solid rgba(99,102,241,.1)}}
.th h2{{font-size:1.2rem;font-weight:600}}
table{{width:100%;border-collapse:collapse}}
th{{text-align:left;padding:1rem 1.5rem;font-size:.75rem;text-transform:uppercase;letter-spacing:.05em;color:#94a3b8;border-bottom:1px solid rgba(99,102,241,.08);font-weight:600}}
td{{padding:1.2rem 1.5rem;border-bottom:1px solid rgba(255,255,255,.03);vertical-align:middle}}
tr:last-child td{{border-bottom:none}}
tr:hover{{background:rgba(99,102,241,.04)}}
.rk{{font-weight:700;color:#6366f1;font-size:1rem}}
.cn{{font-weight:600;font-size:.95rem}}
.sc{{display:flex;align-items:center;gap:.75rem}}
.sb{{flex:1;height:8px;background:rgba(255,255,255,.06);border-radius:999px;overflow:hidden;min-width:80px}}
.sf{{height:100%;border-radius:999px;transition:width 1s ease}}
.tgs{{display:flex;flex-wrap:wrap;gap:.4rem}}
.tag{{font-size:.7rem;padding:.25rem .6rem;border-radius:999px;font-weight:500}}
.tg{{background:rgba(16,185,129,.1);color:#34d399;border:1px solid rgba(16,185,129,.2)}}
.tr{{background:rgba(239,68,68,.1);color:#f87171;border:1px solid rgba(239,68,68,.2)}}
.bd{{display:inline-block;padding:.35rem .9rem;border-radius:999px;font-size:.75rem;font-weight:600;white-space:nowrap}}
.ft{{text-align:center;padding:2rem;color:#475569;font-size:.8rem}}
@keyframes fi{{from{{opacity:0;transform:translateY(10px)}}to{{opacity:1;transform:translateY(0)}}}}
.st,tr{{animation:fi .5s ease forwards}}
.st:nth-child(2){{animation-delay:.1s}}.st:nth-child(3){{animation-delay:.2s}}.st:nth-child(4){{animation-delay:.3s}}.st:nth-child(5){{animation-delay:.4s}}
</style>
</head>
<body>
<div class="ct">
<div class="hd">
<h1>🤖 AI Resume Screening Results</h1>
<p>Powered by GPT-4o-mini • Candidates ranked by AI evaluation score</p>
</div>
<div class="sts">
<div class="st"><div class="sl">Total Candidates</div><div class="sv pr">{len(sorted_df)}</div></div>
<div class="st"><div class="sl">Average Score</div><div class="sv am">{avg_score:.0f}</div></div>
<div class="st"><div class="sl">Top Candidate</div><div class="sv gr" style="font-size:1.3rem">{top_candidate}</div></div>
<div class="st"><div class="sl">Strong Fit</div><div class="sv gr">{strong}</div></div>
<div class="st"><div class="sl">Not Fit</div><div class="sv rd">{not_fit}</div></div>
</div>
<div class="tc">
<div class="th"><h2>📋 Candidate Rankings</h2></div>
<table>
<thead><tr><th>Rank</th><th>Candidate</th><th>Score</th><th>Strengths</th><th>Gaps</th><th>Recommendation</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</div>
<div class="ft">Generated by AI Resume Screening Pipeline</div>
</div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return os.path.abspath(output_path)


# ─── Main Pipeline ──────────────────────────────────────────────
print()
print("╔══════════════════════════════════════════════════════════╗")
print("║          🤖  AI Resume Screening Pipeline  🤖           ║")
print("╚══════════════════════════════════════════════════════════╝")
print()
print(f"✅ API Key loaded (ends with ...{api_key[-4:]})")
print(f"📄 Job Description loaded ({len(jd)} chars)")
print()

# Discover and process resumes
resume_folder = "resumes"
results = []

resume_files = sorted([f for f in os.listdir(resume_folder) if f.endswith(".pdf")])
print(f"📁 Found {len(resume_files)} resume(s)")
print()
print("🔍 Evaluating resumes...")
print("─" * 50)

for i, file in enumerate(resume_files, 1):
    resume_text = extract_text_from_pdf(os.path.join(resume_folder, file))

    # Extract candidate name from file name
    name = file.replace(".pdf", "")
    for line in resume_text.split("\n"):
        if line.startswith("Name:"):
            name = line.replace("Name:", "").strip()
            break

    print(f"  [{i}/{len(resume_files)}] {name}...", end=" ", flush=True)
    result = evaluate_resume(jd, resume_text, name)
    results.append(result)
    print(f"✅ Score: {result['Score']} ({result['Recommendation']})")

print("─" * 50)
print()

# Create DataFrame and rank
df = pd.DataFrame(results)
df = df.sort_values(by="Score", ascending=False).reset_index(drop=True)
df["Rank"] = df["Score"].rank(ascending=False).astype(int)

# Print ranked table
print("📊 FINAL RANKINGS")
print("═" * 80)
print(f"{'Rank':<6}{'Candidate':<20}{'Score':<8}{'Recommendation':<18}{'Top Strength'}")
print("─" * 80)
for _, row in df.iterrows():
    top = row["Strengths"].split(",")[0].strip() if row["Strengths"] else "N/A"
    if len(top) > 25:
        top = top[:22] + "..."
    print(f"#{row['Rank']:<5}{row['Candidate']:<20}{row['Score']:<8}{row['Recommendation']:<18}{top}")
print("═" * 80)
print()

# Save outputs
df.to_csv("output.csv", index=False)
html_path = generate_html_dashboard(df, "output.html")

print(f"💾 Results saved:")
print(f"   📄 CSV:  {os.path.abspath('output.csv')}")
print(f"   🌐 HTML: {html_path}")
print()
print("🎉 Done! Open output.html in your browser for a visual dashboard.")
print("   Or import output.csv into Google Sheets.")
print()
