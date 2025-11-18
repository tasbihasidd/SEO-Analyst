import json
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()


# Initialize LLM
def get_llm():
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")

    llm = ChatOpenAI(
        api_key=api_key,
        base_url="https://api.fireworks.ai/inference/v1",  # Fireworks API endpoint
        model="accounts/fireworks/models/kimi-k2-instruct-0905"
    )
    return llm


SEO_PROMPT = """
You are an expert SEO analyst with 15+ years of experience in technical SEO, on-page SEO, crawling, indexing, content structure, metadata optimization, keyword strategy, and user-intent analysis.
Your task is to analyze the following website SEO data and produce a detailed, professional, client-ready SEO Report.

You will receive JSON input in this exact format:

{
  "pages": [
    {
      "url": "",
      "title": "",
      "description": "",
      "word_count": 0,
      "keywords": [],
      "bigrams": {},
      "trigrams": {},
      "warnings": [],
      "headings": {
        "h2": [],
        "h3": [],
        "h4": []
      },
      "additional_info": {
        "title": [],
        "canonical": [],
        "og_title": [],
        "og_desc": []
      }
    }
  ]
}

Your Tasks
1. HIGH-LEVEL WEBSITE OVERVIEW

Summarize the website’s overall SEO health
Mention strengths, weaknesses, risks
Provide an SEO score (0–100)

2. PAGE-BY-PAGE SEO AUDIT

For each page in the JSON:

Analyze deeply:

🔹 Metadata Analysis

Title quality, length & keyword use
Description quality & missing elements
OG title and OG description consistency
Canonical correctness

🔹 Content Quality

Word count evaluation (short, ideal, long)
Content depth & topical coverage
Keyword density insights (from keywords[])
Bigram & trigram context → topical relevance

🔹 Heading Structure

H2/H3/H4 hierarchy
Missing H1 warnings
Heading keyword alignment

🔹 Technical Issues

Use warnings[] field to detect:

Missing ALT tags
Missing H1
Large images
Broken metadata
Thin content
Canonical issues

3. PRIORITY SEO ISSUES (Critical → Low)

Create a categorized list:
Critical issues
High priority issues
Medium priority
Low priority
Nice-to-have enhancements

4. ACTIONABLE RECOMMENDATIONS

For each page, provide practical step-by-step suggestions:
Exact title rewrite
Exact meta description rewrite
Suggested H1/H2s
Keyword optimization plan
Content expansion suggestions
Technical fixes

5. KEYWORD STRATEGY REPORT

Using:
keywords[]
bigrams
trigrams

Generate:

Top target keywords
Missing but relevant keywords
User intent mapping
Opportunities for ranking improvement
Topic clusters to improve relevance

6. FINAL SUMMARY

A clear, professional summary including:
Overall SEO health
Major blockers
Suggested 30-day action plan
Expected impact of implementing fixes

Tone & Format Requirements

✔ Professional
✔ Data-driven
✔ Very detailed
✔ Bullet points, headings, tables where helpful
✔ Client-friendly, clear, structured
    """

# SEO_PROMPT = """
# You are an SEO expert with 15+ years of experience.
# You MUST generate the SEO report strictly PAGE-BY-PAGE.

# For EVERY page in the JSON, output a separate section in this exact format:

# ------------------------------------------------------------
# PAGE REPORT — {PAGE.URL}

# 1. METADATA ANALYSIS
# - Title:
# - Description:
# - Canonical:
# - OG Title:
# - OG Description:
# → Issues:
# → Recommendations:

# 2. CONTENT QUALITY
# - Word Count:
# - Content Depth Evaluation:
# - Missing Content:

# 3. HEADING STRUCTURE
# - H2:
# - H3:
# - H4:
# → Issues:
# → Recommendations:

# 4. KEYWORD & TOPIC ANALYSIS
# - Top keywords:
# - Bigram patterns:
# - Trigram patterns:
# → What this tells us:
# → Missing keywords:

# 5. TECHNICAL WARNINGS
# (List all warnings)

# 6. PAGE-SPECIFIC ACTION PLAN
# - 3–7 exact actions for this page only
# ------------------------------------------------------------

# RULES:
# - Do NOT merge pages into a summary.
# - Do NOT generalize results.
# - EACH page MUST appear separately in report.
# - Follow the input order exactly.

# After all page reports:
# - Give overall site summary
# - Priority issues
# - 30-day roadmap
# """


# Generate report
def generate_ai_seo_report(data: dict) -> str:
    """
    Sends JSON SEO data + prompt to the AI model and returns the SEO report.
    """
    llm = get_llm()

    full_prompt = SEO_PROMPT + "\n\nHere is the JSON data:\n" + json.dumps(data, indent=2)

    try:
        response = llm.invoke(full_prompt)
        return response.content
    except Exception as e:
        print("Error generating SEO report:", e)
        return ""


def run_seo_report(json_file_path: str, output_txt_file: str):
    """
    Loads optimized JSON → sends to AI → saves full SEO report as a text file.
    """

    # Load your optimized SEO JSON
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("Sending data to AI for SEO analysis...")

    report = generate_ai_seo_report(data)

    # Save final report as text/markdown
    with open(output_txt_file, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nSEO Report successfully generated: {output_txt_file}")

# Run
if __name__ == "__main__":
    run_seo_report(
        json_file_path="seo_report_optimized.json",
        output_txt_file="FINAL_SEO_REPORT.txt"
    )