import streamlit as st
import google.generativeai as genai
from datetime import date
import os

# Set Gemini API Key
GOOGLE_API_KEY = "AIzaSyAx4bAdsO3o41eCGiyKiZSgPjlPhNxNH9g"  # Replace with your actual key
genai.configure(api_key=GOOGLE_API_KEY)

# Load Gemini Model
model = genai.GenerativeModel("gemini-2.0-flash-lite")

# UI setup
st.set_page_config(page_title="Gemini Trending Post Generator", page_icon="📊")
st.title("🔥 Trending Post Generator with Gemini")

# Inputs
col1, col2 = st.columns(2)
selected_date = col1.date_input("Select Date", value=date.today())
platform = col2.selectbox("Choose Platform", ["LinkedIn", "Instagram", "YouTube", "Twitter", "Threads"])
topic = st.text_input("Enter a Topic", placeholder="e.g. AI in Education")

# Generate button
if st.button("Generate Trending Posts"):
    if not topic.strip():
        st.warning("Please enter a topic.")
    else:
        prompt = f"""
        You are a social media strategist. Imagine it's {selected_date.strftime('%B %d, %Y')}.
        Based on the topic "{topic}" and the platform "{platform}", generate 3 trending post ideas.

        For each post:
        - A short title
        - A one-liner caption
        - 3-5 relevant, catchy hashtags

        Present them clearly and attractively.
        """

        try:
            response = model.generate_content(prompt)
            st.success("Generated trending post ideas successfully!")
            st.markdown("### 📌 Suggested Posts")
            st.write(response.text)
        except Exception as e:
            st.error(f"Something went wrong: {e}")
