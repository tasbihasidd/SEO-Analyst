import streamlit as st
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
from google import genai
from google.genai import types

load_dotenv()

GEMINI_KEY = st.secrets["GEMINI_API_KEY"]
OPENAI_KEY = st.secrets["OPENAI_API_KEY"]

# ======================================================
# GOOGLE IMAGEN 4 — SAFE IMAGE GENERATOR
# ======================================================
def generate_image(prompt):
    try:
        client = genai.Client(api_key=GEMINI_KEY)

        response = client.models.generate_images(
            model="imagen-4.0-generate-001",
            prompt=prompt,
            config=types.GenerateImagesConfig(number_of_images=1)
        )

        img_obj = response.generated_images[0].image   # google.genai.types.Image object

        # -------- Correct handling for google.genai.types.Image --------

        # Case 1: If Google already built a PIL Image internally
        if hasattr(img_obj, "_pil_image") and isinstance(img_obj._pil_image, Image.Image):
            return img_obj._pil_image

        # Case 2: If raw image bytes exist
        if hasattr(img_obj, "image_bytes") and img_obj.image_bytes:
            return Image.open(BytesIO(img_obj.image_bytes))

        # Case 3: Try .save() API
        if hasattr(img_obj, "save"):
            buffer = BytesIO()
            img_obj.save(buffer, format="PNG")
            buffer.seek(0)
            return Image.open(buffer)

        raise ValueError("Unable to extract image from google.genai.types.Image")

    except Exception as e:
        print("Image generation error:", e)
        return None



# ======================================================
# FIREWORKS LLM SETUP
# ======================================================
def get_llm():
    if not OPENAI_KEY:
        raise ValueError("OPENAI_API_KEY not found.")

    return ChatOpenAI(
        api_key=OPENAI_KEY,
        base_url="https://api.fireworks.ai/inference/v1",
        model="accounts/fireworks/models/gpt-oss-120b"
    )


# ======================================================
# STREAMLIT UI
# ======================================================
st.set_page_config(page_title="Content Generator", layout="wide")
st.title("📝 AI Content Generation App")

company_overview = st.text_area("🏢 Company Overview", height=150)
category = st.text_input("📂 Category")

if "suggested_topics" not in st.session_state:
    st.session_state.suggested_topics = []


# ======================================================
# TOPIC GENERATOR
# ======================================================
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
            - Short and scroll-stopping.
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


# ======================================================
# PROMPT BUILDER
# ======================================================
def build_prompt(category, topic, platform, post_length, overview):
    return f"""
    You are an expert social media content creator.
    Generate a highly engaging, platform-optimized post.

    Category: {category}
    Topic: {topic}
    Platform: {platform}
    Post Length: {post_length}
    Company Overview: {overview}

    Follow platform rules strictly.
    """


# ======================================================
# CONTENT + IMAGE GENERATION
# ======================================================
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

            # ===================== IMAGE GENERATION =====================
            image_prompt = f"""
            Create a modern, aesthetic, scroll-stopping social media image
            inspired by this post content:

            \"\"\"{result}\"\"\"

            Requirements:
            - Match emotional tone of the post.
            - Style must match platform: {platform}
            - Clean, modern, professional.
            - No text inside image.
            """

            with st.spinner("Generating AI Image..."):
                img = generate_image(image_prompt)

            if img:
                st.image(img, caption="AI Generated Image", use_column_width=True)

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
