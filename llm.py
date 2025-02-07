import streamlit as st
import time
import warnings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA
from decouple import config

# Suppress warnings
warnings.filterwarnings("ignore")

# Streamlit App Title
st.title("Chatbot with RAG - PDF Question Answering")

# Google API Key
GOOGLE_API_KEY = config("GOOGLE_API_KEY")

# Initialize LLM Model
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
    convert_system_message_to_human=True,
    google_api_key=GOOGLE_API_KEY
)

# Function to Stream Text Output
def stream_data(response):
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.02)

# Upload PDF File
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    st.success("PDF uploaded successfully! Processing...")
    
    # Save Uploaded File Temporarily
    pdf_path = f"./uploaded_{uploaded_file.name}"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load and Process PDF
    pdf_loader = PyPDFLoader(pdf_path)
    pages = pdf_loader.load_and_split()
    
    # Text Splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    context = "\n\n".join(str(p.page_content) for p in pages)
    texts = text_splitter.split_text(context)

    # Embeddings Model
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    # Create a New Chroma Vector Store for Each PDF
    vector_store = Chroma.from_texts(texts, embeddings, persist_directory="./chroma_db")
    vector_store.persist()

    # Create Retriever
    vector_index = vector_store.as_retriever(search_kwargs={"k": 5})

    # Create Retrieval-QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        model,
        retriever=vector_index,
        return_source_documents=True
    )

    # User Query
    user_input = st.text_input("Ask a question about the uploaded PDF:")
    
    # Function to clean the output
    def clean_response(response):
        response = response.replace("<sup>", "^").replace("</sup>", "")
        response = response.replace("<sub>", "~").replace("</sub>", "")
        return response

    if user_input:
        with st.spinner("Processing..."):
            response_placeholder = st.empty()
            streamed_text = ""

            # Query the model
            result = qa_chain({"query": user_input})

            # Clean and display response
            cleaned_response = clean_response(result["result"])
            response_placeholder.markdown(cleaned_response, unsafe_allow_html=True)
