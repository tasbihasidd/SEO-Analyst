import streamlit as st
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from openai import OpenAI
import base64
from PIL import Image
from io import BytesIO

load_dotenv()

os.getenv("GEMINI_API_KEY")
# =============== GOOGLE IMAGEN 4 IMAGE GENERATION ===============
def generate_image(prompt):
    try:
        client = OpenAI(
        api_key="GEMINI_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

        response = client.images.generate(
            model="imagen-3.0-generate-002",
            prompt=prompt,
            response_format='b64_json',
            n=1,
        )

        for image_data in response.data:
            image = Image.open(BytesIO(base64.b64decode(image_data.b64_json)))
            image.show()

    except Exception as e:
        print("Image generation error:", e)
        return None


# =============== FIREWORKS LLM SETUP ===============
def get_llm():
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY not found.")

    llm = ChatOpenAI(
        api_key=api_key,
        base_url="https://api.fireworks.ai/inference/v1",
        model="accounts/fireworks/models/gpt-oss-120b"
    )
    return llm


# =============== STREAMLIT UI ===============
st.set_page_config(page_title="Content Generator", layout="wide")
st.title("📝 AI Content Generation App")

company_overview = st.text_area("🏢 Company Overview", height=150)
category = st.text_input("📂 Category")

if "suggested_topics" not in st.session_state:
    st.session_state.suggested_topics = []


# =============== GENERATE TOPICS ===============
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
            """

            response = llm.invoke(prompt)
            raw_topics = response.content.split("\n")

            cleaned = []
            for t in raw_topics:
                t = t.strip().lstrip("-•0123456789. ").strip()
                if len(t) > 3:
                    cleaned.append(t)

            st.session_state.suggested_topics = cleaned[:10]

        except Exception as e:
            st.error(f"Topic generation failed: {e}")


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


# =============== PROMPT BUILDER ===============
def build_prompt(category, topic, platform, post_length, overview):
    return f"""
    You are an expert social media content creator.
    Generate a highly engaging, platform-optimized post.

    Category: {category}
    Topic: {topic}
    Platform: {platform}
    Post Length: {post_length}
    Company Overview: {overview}

    Follow all platform style rules strictly.
    """


# =============== CONTENT + IMAGE GENERATION ===============
if generate:
    if not category or not selected_topic or not company_overview:
        st.error("⚠️ All fields are required!")
    else:
        topic_str = ", ".join(selected_topic)

        try:
            llm = get_llm()

            with st.spinner("Generating post content..."):
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


            # ---------- IMAGE GENERATION BASED ON POST CONTENT ----------
            image_prompt = f"""
            Create a visually appealing, scroll-stopping social media image
            inspired by the following post content:

            \"\"\"{result}\"\"\"

            Requirements:
            - Match the tone & emotion of the post.
            - Follow the platform style: {platform}.
            - Aesthetic, modern, professional.
            - No text in the image.
            """

            with st.spinner("Generating AI Image..."):
                img = generate_image(image_prompt)

            if img:
                st.image(img, caption="AI Generated Post Image", use_column_width=True)

                buffer = BytesIO()
                img.save(buffer, format="PNG")
                img_bytes = buffer.getvalue()

                st.download_button(
                    label="⬇️ Download Image",
                    data=img_bytes,
                    file_name=f"{topic_str[:30].replace(' ','_')}_image.png",
                    mime="image/png"
                )
            else:
                st.error("❌ Image generation failed. Check logs.")

        except Exception as e:
            st.error(f"❌ Error: {e}")
