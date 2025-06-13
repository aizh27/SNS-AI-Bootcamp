import os
import streamlit as st
from langchain.chat_models import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser

# Set up the Gemini API key
# Option 1: via environment variable (recommended for local dev)
os.environ["GOOGLE_API_KEY"] = st.secrets["google"]["AIzaSyBHRfQLiGDN0zf-c_wzMoIcdsezlyIDWWg"] if "google" in st.secrets else "YOUR_GEMINI_API_KEY"

# Initialize the Gemini Chat model using LangChain
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.environ["GOOGLE_API_KEY"])

# Define the ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful translator that translates English to French."),
    ("user", "Translate this sentence to French: {sentence}")
])

# Create a chain: prompt -> llm -> parse output
chain: Runnable = prompt | llm | StrOutputParser()

# Streamlit App UI
st.set_page_config(page_title="English to French Translator", layout="centered")
st.title("🌍 English to French Translator")

# Input area
sentence = st.text_input("Enter an English sentence to translate:")

# Translate button
if st.button("Translate"):
    if not sentence.strip():
        st.warning("Please enter a sentence.")
    else:
        try:
            # Run the chain with input
            with st.spinner("Translating..."):
                result = chain.invoke({"sentence": sentence})
            # Display translation
            st.success("Translation successful!")
            st.markdown(f"**French Translation:** {result}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
