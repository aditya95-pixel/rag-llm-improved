import streamlit as st
import time
import warnings
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA
from rake_nltk import Rake
from decouple import config

def stream_data(response):
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.02)

warnings.filterwarnings("ignore")

st.title("Chatbot with RAG - Document Question Answering")

GOOGLE_API_KEY = config("GOOGLE_API_KEY")

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
    convert_system_message_to_human=True,
    google_api_key=GOOGLE_API_KEY
)

CHROMA_DB_DIR = "./chroma_db"

def clear_chroma_content():
    if os.path.exists(CHROMA_DB_DIR):
        try:
            vector_store = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=GoogleGenerativeAIEmbeddings(
                model="models/embedding-001", google_api_key=GOOGLE_API_KEY))
            vector_store.delete_collection()
            del vector_store
        except Exception as e:
            st.warning(f"Could not clear ChromaDB content: {e}")

uploaded_file = st.file_uploader("Upload a PDF, Word, or TXT file", type=["pdf", "docx", "txt"])

if uploaded_file:
    st.success("File uploaded successfully! Processing...")

    clear_chroma_content()

    file_path = f"./uploaded_{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if uploaded_file.type == "application/pdf":
        loader = PyPDFLoader(file_path)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        loader = Docx2txtLoader(file_path)
    elif uploaded_file.type == "text/plain":
        loader = TextLoader(file_path)
    else:
        st.error("Unsupported file type.")
        st.stop()
    
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    context = "\n\n".join(str(p.page_content) for p in pages)
    texts = text_splitter.split_text(context)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    vector_store = Chroma.from_texts(texts, embeddings, persist_directory=CHROMA_DB_DIR)
    vector_store.persist()

    vector_store._collection = None
    del vector_store

    vector_index = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings).as_retriever(search_kwargs={"k": 5})

    qa_chain = RetrievalQA.from_chain_type(
        model,
        retriever=vector_index,
        return_source_documents=True
    )
    def clean_response(response):
        response = response.replace("<sup>", "^").replace("</sup>", "")
        response = response.replace("<sub>", "~").replace("</sub>", "")
        return response
    
    def suggest_queries(text):
        rake = Rake()
        rake.extract_keywords_from_text(text)
        return rake.get_ranked_phrases()[:5] 

    query_suggestions = suggest_queries(context)

    st.sidebar.markdown("### 🔍 Suggested Queries")
    clicked_query = None
    for qs in query_suggestions:
        if st.sidebar.button(qs):  
            clicked_query = qs  

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Ask a question about the document:", value=clicked_query if clicked_query else "")

    if user_input:
        with st.spinner("Processing..."):
            response_placeholder = st.empty()
            result = qa_chain({"query": user_input})
            cleaned_response = clean_response(result["result"])
            
            st.session_state.chat_history.append((user_input, cleaned_response))

            streamed_text = ""
            for word in stream_data(cleaned_response):
                streamed_text += word
                response_placeholder.markdown(streamed_text)

    st.sidebar.markdown("### 💬 Chat History")
    for query, response in reversed(st.session_state.chat_history):
        with st.sidebar.expander(query):
            st.markdown(response)

