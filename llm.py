import streamlit as st #python library for User Interface 
import time #python library for streamed response (time gaps)
import warnings #python library for handling warnings
import os #python library for dealing with directories,files,and environment variables
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings 
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA
from decouple import config

#generates string one by one with a delay of 20 milisecond
def stream_data(response):
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.02)

#prevents warnings from being displayed
warnings.filterwarnings("ignore")

#main title
st.title("Chatbot with RAG - PDF Question Answering")

#config function is used to load Google API key from configuration file
GOOGLE_API_KEY = config("GOOGLE_API_KEY")

#initialize Google gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", #model name
    temperature=0.2, #low randomness
    convert_system_message_to_human=True, #useful for system instructions
    google_api_key=GOOGLE_API_KEY #plugging in the API key
)

#chroma db directory for in memory storage
CHROMA_DB_DIR = "./chroma_db"

# Function to clear ChromaDB content before new document processing
def clear_chroma_content():
    if os.path.exists(CHROMA_DB_DIR):
        try:
            vector_store = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=GoogleGenerativeAIEmbeddings(
                model="models/embedding-001", google_api_key=GOOGLE_API_KEY))
            vector_store.delete_collection() # Deletes existing collection to refresh the database
            del vector_store
        except Exception as e:
            st.warning(f"Could not clear ChromaDB content: {e}")

#to upload a pdf
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    st.success("PDF uploaded successfully! Processing...")

    clear_chroma_content() # Clear old data before storing new document

    #updated file saved locally
    pdf_path = f"./uploaded_{uploaded_file.name}"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load the PDF and split it into pages
    pdf_loader = PyPDFLoader(pdf_path)
    pages = pdf_loader.load_and_split()

    # Split text into smaller chunks for vector storage
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    context = "\n\n".join(str(p.page_content) for p in pages)
    texts = text_splitter.split_text(context)
    
    # Initialize embeddings using Google Generative AI
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    # Create a vector store using ChromaDB and save the document
    vector_store = Chroma.from_texts(texts, embeddings, persist_directory=CHROMA_DB_DIR)
    vector_store.persist()

    # Clear memory usage of the vector store
    vector_store._collection = None 
    del vector_store  

    # Retrieve relevant chunks (top 5) from the vector store
    vector_index = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings).as_retriever(search_kwargs={"k": 5})

    # Create a QA chain that retrieves answers from stored knowledge
    qa_chain = RetrievalQA.from_chain_type(
        model,
        retriever=vector_index,
        return_source_documents=True
    )

    # User input field for asking questions about the PDF
    user_input = st.text_input("Ask a question about the uploaded PDF:")

    # Function to clean response text before displaying
    def clean_response(response):
        response = response.replace("<sup>", "^").replace("</sup>", "")
        response = response.replace("<sub>", "~").replace("</sub>", "")
        return response

    # If user provides input, process the query
    if user_input:
        with st.spinner("Processing..."):
            response_placeholder = st.empty()
            # Get response from the QA chain
            result = qa_chain({"query": user_input})

            # Clean the response for proper formatting
            cleaned_response = clean_response(result["result"])
            streamed_text = "" 
            for word in stream_data(result["result"]):
                streamed_text += word
                response_placeholder.markdown(streamed_text)
