from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
import warnings
from langchain.chains.question_answering.chain import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA 
from decouple import config
import streamlit as st
import time
def stream_data(response):
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.02)
st.title("Chatbot with RAG")
warnings.filterwarnings("ignore")
GOOGLE_API_KEY = config("GOOGLE_API_KEY") 
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash",temperature=0.2,convert_system_message_to_human=True,google_api_key=GOOGLE_API_KEY)
pdf_loader=PyPDFLoader("attention_is_all_you_need.pdf")
pages=pdf_loader.load_and_split()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
context="\n\n".join(str(p.page_content) for p in pages)
texts=text_splitter.split_text(context)
embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=GOOGLE_API_KEY) #1
vector_store=Chroma.from_texts(texts,embeddings,persist_directory="./chroma_db")
vector_store.persist()
vector_index=vector_store.as_retriever(search_kwargs={"k":5})
qa_chain=RetrievalQA.from_chain_type(
    model,
    retriever=vector_index,
    return_source_documents=True
)
user_input = st.text_input("Enter your query:")
result=qa_chain({"query":user_input})
if user_input:
    with st.spinner("Processing..."):
        response_placeholder = st.empty()
        streamed_text = "" 
        for word in stream_data(result["result"]):
            streamed_text += word
            response_placeholder.markdown(streamed_text)