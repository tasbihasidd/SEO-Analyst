import streamlit as st
import json
from pyseoanalyzer import analyze
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import time

load_dotenv()

# ---------------------------
#  LLM SETUP
# ---------------------------
def get_llm():
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY missing in .env")

    llm = ChatOpenAI(
        api_key=api_key,
        base_url="https://api.fireworks.ai/inference/v1",
        model="accounts/fireworks/models/kimi-k2-instruct-0905"
    )
    return llm


# ---------------------------
#  PROMPT
# ---------------------------
SEO_PROMPT = """
You are an expert SEO analyst with 15+ years of experience.
Generate a detailed, client-ready SEO audit based on the SEO JSON below.

Follow ALL structure exactly as instructed earlier.

Here is the JSON data:
"""

# ---------------------------
#  FUNCTIONS
# ---------------------------

def generate_seo_json(url: str):
    """Runs pyseoanalyzer and returns JSON report."""
    return analyze(
        url,
        follow_links=True,
        analyze_headings=True,
        analyze_extra_tags=True
    )


def generate_ai_seo_report(data: dict) -> str:
    """Sends SEO JSON + prompt to AI."""
    llm = get_llm()
    full_prompt = SEO_PROMPT + json.dumps(data, indent=2)

    try:
        response = llm.invoke(full_prompt)
        return response.content
    except Exception as e:
        return f"Error generating AI report: {e}"


# ---------------------------
#  STREAMLIT UI
# ---------------------------

st.set_page_config(page_title="AI SEO Analyzer", layout="wide")

st.title("🔍 AI SEO Analyzer – Full Website Audit")
st.write("Provide a URL below and generate a complete SEO report.")

# URL INPUT
website_url = st.text_input("Enter Website URL:", placeholder="https://example.com")

# Run Button
if st.button("Generate SEO Report"):
    if not website_url.strip():
        st.error("❌ Please enter a valid website URL.")
        st.stop()

    st.info("⏳ Crawling website & analyzing SEO… this may take 20–60 seconds.")
    with st.spinner("Running analysis…"):

        # STEP 1 – Extract SEO JSON
        try:
            seo_json = generate_seo_json(website_url)
        except Exception as e:
            st.error(f"❌ Error analyzing site: {e}")
            st.stop()

        st.success("✔ SEO JSON generated.")

        # Show preview
        st.subheader("📄 SEO JSON Preview")
        st.json(seo_json)

        # STEP 2 – Generate AI Report
        st.info("🤖 Generating AI SEO Report…")
        time.sleep(1)

        ai_report = generate_ai_seo_report(seo_json)

        st.success("✔ AI SEO Report generated!")

        # Show final SEO report
        st.subheader("📘 Final SEO Report")
        st.write(ai_report)

        # -------------------------
        # DOWNLOAD BUTTONS
        # -------------------------

        # JSON file
        json_bytes = json.dumps(seo_json, indent=4).encode("utf-8")
        st.download_button(
            label="⬇ Download SEO JSON",
            data=json_bytes,
            file_name="seo_report.json",
            mime="application/json"
        )

        # AI SEO Report file
        txt_bytes = ai_report.encode("utf-8")
        st.download_button(
            label="⬇ Download AI SEO Report",
            data=txt_bytes,
            file_name="FINAL_SEO_REPORT.txt",
            mime="text/plain"
        )
