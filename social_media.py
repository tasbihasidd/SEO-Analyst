import streamlit as st
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
import google.generativeai as gen

load_dotenv()

# --- Setup ---
def get_llm():
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")

    llm = ChatOpenAI(
        api_key=api_key,
        base_url="https://api.fireworks.ai/inference/v1",
        model="accounts/fireworks/models/gpt-oss-120b"
    )
    return llm


st.set_page_config(page_title="Content Generator", layout="wide")
st.title("📝 AI Content Generation App")

# --- Inputs ---
company_overview = st.text_area("🏢 Company Overview", height=150)
category = st.text_input("📂 Category")

# SESSION STATE for stable topics
if "suggested_topics" not in st.session_state:
    st.session_state.suggested_topics = []

# Button to generate topics
if st.button("🔍 Generate Topics"):
    if not category:
        st.error("Please enter a category first!")
    else:
        try:
            llm = get_llm()

            prompt = f"""
            Generate 10 unique, modern, engaging content topics for category:
            "{category}"

            Rules:
            - No numbering.
            - Short, clean, scroll-stopping topics.
            - Avoid generic ideas.
            - Make every generation 100% different.
            """

            response = llm.invoke(prompt)
            raw_topics = response.content.split("\n")

            # Clean topics (remove numbering, extra symbols)
            cleaned = []
            for t in raw_topics:
                t = t.strip().lstrip("-•0123456789. ").strip()
                if len(t) > 3:
                    cleaned.append(t)

            st.session_state.suggested_topics = cleaned[:10]  # ensure max 10

        except Exception as e:
            st.error(f"Topic generation failed: {e}")

# MULTISELECT for choosing topics
selected_topic = st.multiselect(
    "🎯 Choose from Suggested Topics",
    st.session_state.suggested_topics
)

platform = st.selectbox(
    "📱 Social Media Platform",
    ["Facebook", "Instagram", "LinkedIn"]
)

post_length = st.selectbox(
    "✏️ Post Length",
    ["Short", "Medium", "Long"]
)

generate = st.button("Generate Content")


# --- Prompt Template ---
def build_prompt(category, topic, platform, post_length, overview):
    return f"""
    You are an expert social media content creator.
    Generate a highly engaging, platform-optimized post.

    Inputs:

    Category: {category}

    Topic: {topic}

    Platform: {platform}

    Post Length: {post_length}

    Company Overview: {overview}

    Platform Rules (Very Important — Follow Strictly)
    1. LinkedIn Rules

    Tone: Professional, expert, industry-insightful.

    Structure:
    Strong hook (insight/statistic/problem).
    2–3 short paragraphs.
    Optional: Expertise statement about the company.
    Soft call-to-action.

    Length Guidance:
    Short → 1–2 sharp professional lines
    Medium → 4–6 informative lines
    Long → Mini storytelling + value points

    Headings/Subheadings:
    Allowed (Example: “Key Insights”, “Here’s How It Helps”)

    Formatting:
    No emojis OR minimal (1–2 max)
    Max 4 hashtags
    No line breaks spam

    Goal: Establish authority & expertise.

    2. Facebook Rules
    Tone: Friendly, conversational, easy-to-understand.

    Structure:
    Relatable hook
    Problem + solution
    Light promotion
    Clear CTA

    Length Guidance:
    Short → 1–2 conversational lines
    Medium → 3–5 lines in a friendly tone
    Long → Story-style with simple language

    Headings/Subheadings:
    Optional but simple

    Formatting:
    Emojis allowed
    8–12 hashtags

    Goal: Engagement + user interaction.

    3. Instagram Rules
    Tone: Short, trendy, emotional, aesthetic.

    Structure:
    One-line punchy hook
    2–3 short bullet points or lines
    Strong CTA (DM / Follow / Save)

    Length Guidance:

    Short → 1 aesthetic line
    Medium → 3–5 lines with spacing
    Long → Short storytelling but still compact

    Headings/Subheadings:
    Not required (IG prefers clean captions)

    Formatting:

    Emojis allowed
    Line breaks
    10–15 hashtags

    Goal: Aesthetic engagement + saves + shares.

    Common Requirements for All Platforms
    Auto-adjust tone based on selected platform.

    Word count must follow selected post length:
    Short → 1–2 lines
    Medium → 3–5 lines
    Long → Storytelling style

    Auto-generate hashtags:
    LinkedIn: Max 4
    Facebook: 8–12
    Instagram: 10–15

    Use inputs naturally within the content.

    Content must be original, engaging, and platform-optimized.
    ---
    """


# --- Generation ---
if generate:
    if not category or not selected_topic or not company_overview:
        st.error("⚠️ Please fill all fields.")
    else:
        topic_str = ", ".join(selected_topic)  # multiple topics supported

        try:
            llm = get_llm()

            with st.spinner("Generating content..."):
                response = llm.invoke(
                    build_prompt(category, topic_str, platform, post_length, company_overview)
                )

            result = response.content
            
            st.success("Content Generated Successfully!")
            st.write("---")
            st.markdown(result)

            st.download_button(
                label="⬇️ Download Content",
                data=result,
                file_name=f"{topic_str[:30].replace(' ','_')}_{platform}_post.txt"
            )

        except Exception as e:
            st.error(f"❌ Error: {e}")
