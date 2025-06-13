import streamlit as st
import os
from langchain.chat_models import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

# --------------------------------------------
# 1. Hardcoded Gemini API key and model setup
# --------------------------------------------
os.environ["GOOGLE_API_KEY"] = "AIzaSyBHRfQLiGDN0zf-c_wzMoIcdsezlyIDWWg"

# Initialize Gemini LLM with specified model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

# --------------------------------------------
# 2. Prompt Template for Translation
# --------------------------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that translates English sentences to French."),
    ("user", "Translate this sentence to French: {sentence}")
])

# --------------------------------------------
# 3. Create LangChain Runnable Chain
# --------------------------------------------
chain: Runnable = prompt | llm | StrOutputParser()

# --------------------------------------------
# 4. Streamlit UI
# --------------------------------------------
st.set_page_config(page_title="English to French Translator", layout="centered")
st.title("🌍 English to French Translator")

# Input field
sentence = st.text_input("Enter an English sentence:")

# Translate button
if st.button("Translate"):
    if not sentence.strip():
        st.warning("Please enter a sentence to translate.")
    else:
        try:
            with st.spinner("Translating..."):
                result = chain.invoke({"sentence": sentence})
            st.success("Translation successful!")
            st.markdown(f"**🇫🇷 French Translation:** `{result.strip()}`")
        except Exception as e:
            st.error(f"An error occurred: {e}")
