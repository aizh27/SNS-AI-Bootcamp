import streamlit as st
import tempfile
import google.generativeai as genai
import docx2txt
from pypdf import PdfReader

# ✅ API Key (do NOT use list format!)
genai.configure(api_key="AIzaSyABaQI5JNaveD56SLcLTsqOmsdqpseMzsk")

# ✅ Streamlit UI Setup
st.set_page_config(page_title="📄 Gemini Q&A from Document", layout="wide")
st.title("📄 Ask Questions from Your Document")

uploaded_file = st.file_uploader("📎 Upload your PDF or DOCX file", type=["pdf", "docx"])
question = st.text_input("❓ Ask a question based on the document", placeholder="e.g. What is the summary of chapter 2?")

if uploaded_file and question:
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
        tmp.write(uploaded_file.getvalue())
        path = tmp.name

    try:
        # ✅ Extract text from PDF or DOCX
        if uploaded_file.name.endswith(".pdf"):
            reader = PdfReader(path)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
        else:
            text = docx2txt.process(path)

        # ✅ Setup Gemini model
        model = genai.GenerativeModel("gemini-1.5-flash")

        # ✅ Prompt
        prompt = f"""
You are an intelligent assistant helping users extract knowledge from documents.
Use the content below to answer the user's question.

Document Content:
{text}

User's Question:
{question}

Answer:
"""

        # ✅ Generate Answer
        with st.spinner("Generating answer with Gemini..."):
            response = model.generate_content(prompt)
            st.success("✅ Here's the answer:")
            st.write(response.text)

    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")
Collapse

















