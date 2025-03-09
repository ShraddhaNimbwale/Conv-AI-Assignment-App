pip install chromadb

!pip install faiss-cpu
!pip install transformers
pip install --upgrade protobuf transformers tensorflow
pip install --upgrade langchain langchain-community
pip install rank-bm25


pip install pypdf


pip install sentence-transformers

pip install tf-keras

pip install --upgrade jupyterlab ipywidgets


import streamlit as st
import os
import chromadb
import faiss
import numpy as np
import pandas as pd
import json
import re
import nltk
from nltk.corpus import stopwords

from collections import deque
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

DATA_PATH = "./"
DATA_FILES = ["INFY_2022_2023.pdf", "INFY_2023_2024.pdf"]
text_chunks = []

# Clean Input Data
def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    words = text.lower().split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Load and chunk documents
def load_data():
    global text_chunks
    text_chunks.clear()
    for file in DATA_FILES:
        loader = PyPDFLoader(os.path.join(DATA_PATH, file))
        pages = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        text_chunks.extend(text_splitter.split_documents(pages))

load_data()

# Embeddings and Vector Store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma.from_documents(text_chunks, embedding_model)

# Keyword Retrieval using BM25
bm25_corpus = [chunk.page_content for chunk in text_chunks]
bm25_tokenized = [doc.split() for doc in bm25_corpus]
bm25_model = BM25Okapi(bm25_tokenized)

# Memory-Augmented Retrieval
class MAR:
    def __init__(self, max_memory_size=5):
        self.memory = deque(maxlen=max_memory_size)

    def store_interaction(self, query, response):
        self.memory.append((query, response))

    def retrieve_memory(self):
        return "\n".join([f"User: {q}\nBot: {r}" for q, r in self.memory])

memory_retriever = MAR(max_memory_size=5)

def retrieve_context(query, top_k=3):
    embedding_retrieval = vector_db.similarity_search(query, k=top_k)
    bm25_retrieval = bm25_model.get_top_n(query.split(), bm25_corpus, n=top_k)
    memory_context = memory_retriever.retrieve_memory()
    return "\n".join([doc.page_content for doc in embedding_retrieval] + bm25_retrieval + [memory_context])

# Load Small Language Model (SLM)
MODEL_NAME = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)

# Input Guardrail
def filter_query(query):
    blacklist = {"politics", "sports", "entertainment", "religion", "violence", "hacking", "fraud", "leak"}
    return ("I'm sorry, but I can only answer financial-related questions.", 0.0) if any(word in query.lower() for word in blacklist) else None

# Confidence Score Calculation
def compute_confidence_score(query, response, retrieved_context):
    if not retrieved_context.strip():
        return 0.0

    response_embedding = embedding_model.embed_query(response)
    context_embedding = embedding_model.embed_query(retrieved_context)
    query_embedding = embedding_model.embed_query(query)

    if None in (response_embedding, context_embedding, query_embedding):
        return 0.0

    response_similarity = cosine_similarity([response_embedding], [context_embedding])[0][0]
    query_similarity = cosine_similarity([query_embedding], [context_embedding])[0][0]
    confidence_score = round((0.7 * query_similarity) + (0.3 * response_similarity), 2)
    return max(0.0, min(confidence_score, 1.0)) if confidence_score >= 0.4 else 0.0

# Output Guardrail
def filter_response(response):
    blacklist_phrases = {"I don't know", "I'm not sure", "uncertain", "possibly", "maybe", "likely", "hypothetically", "guess", "it seems", "could be"}
    if any(phrase in response for phrase in blacklist_phrases):
        return "I'm sorry, but I couldn't find a reliable answer based on the financial data provided."
    return "\n".join(list(dict.fromkeys(response.split("\n"))))

# Generate Response
def generate_response(query):
    filtered = filter_query(query)
    if filtered:
        return filtered

    context = retrieve_context(query)
    prompt = f"Memory:\n{context}\n\nUser Query: {query}\n\nAnswer:"
    response = generator(prompt)[0]['generated_text']
    confidence_score = compute_confidence_score(query, response, context)
    
    memory_retriever.store_interaction(query, response)
    return response, confidence_score

# Streamlit UI
st.title("📊 RAG Chatbot - Financial Insights")
st.write("Ask questions about financial statements!")

query = st.text_input("Enter your financial question:")
if st.button("Get Answer"):
    if query:
        response, confidence = generate_response(query)
        st.subheader("Response:")
        st.write(response)
        st.subheader("Confidence Score:")
        st.write(confidence)
