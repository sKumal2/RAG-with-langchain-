# vectorstore.py
from dotenv import load_dotenv
import os
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pymongo import MongoClient

load_dotenv()  

# All the shared stuff lives here
MONGODB_URI = os.getenv("MONGODB_URI")
ATLAS_VECTOR_SEARCH_INDEX_NAME = os.getenv("ATLAS_VECTOR_SEARCH_INDEX_NAME")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

client = MongoClient(MONGODB_URI)
collection = client["rag_db"]["lilianweng_agents"]

vector_store = MongoDBAtlasVectorSearch(
    embedding=embeddings,
    collection=collection,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    relevance_score_fn="cosine",
)

print(f"✓ Connected to MongoDB Atlas vector store ({collection.count_documents({})} chunks)")