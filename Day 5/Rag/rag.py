import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Temporary security compromise ---
GOOGLE_API_KEY = "AIzaSyDDANK56dFae3szwkTz5244asYXvD4fykc"  # Replace with your API key

# --- UI Header ---
st.title("📄 Gemini RAG App with LangChain")
uploaded_file = st.file_uploader("Upload a PDF Document", type="pdf")
query = st.text_input("Ask a question based on the document:")

# --- Gemini LLM Setup ---
llm = ChatGoogleGenerativeAI(
    model="Gemini 2.0 Flash",
    temperature=0.3,
    google_api_key=GOOGLE_API_KEY
)

# --- If File is Uploaded ---
if uploaded_file and query:
    # Load PDF with LangChain
    loader = PyPDFLoader(uploaded_file.name)
    docs = loader.load()

    # Chunking the document
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # Embed and store using FAISS
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Perform similarity search
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Prompt Template
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Use the following context to answer the question:\n\n{context}\n\nQuestion: {question}"
    )

    # RAG Chain setup
    chain = (
        RunnableMap({
            "context": lambda x: retriever.invoke(x["question"]),
            "question": RunnablePassthrough()
        })
        | prompt
        | llm
    )

    # Run the chain
    with st.spinner("Generating answer..."):
        result = chain.invoke({"question": query})

    # Display answer
    st.subheader("Answer:")
    st.write(result.content)

    # Optional: Show retrieved docs
    with st.expander("See Retrieved Chunks"):
        for i, doc in enumerate(retriever.invoke(query)):
            st.markdown(f"**Chunk {i+1}:** {doc.page_content[:500]}...")

