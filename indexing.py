import getpass
import os
from dotenv import load_dotenv
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pymongo import MongoClient


load_dotenv()

MONGODB_URI = os.environ.get("MONGODB_URI")
ATLAS_VECTOR_SEARCH_INDEX_NAME = os.environ.get("ATLAS_VECTOR_SEARCH_INDEX_NAME")

#document embedding
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")


embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
# Only keep post title, headers, and content from the full HTML.
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))

#mongodb atlas connection
client = MongoClient(MONGODB_URI)
MONGODB_COLLECTION = client["rag_db"]["lilianweng_agents"]   # db_name.collection_name – change if you want

#vector store using mongodb
vector_store = MongoDBAtlasVectorSearch(
    embedding=embeddings,
    collection=MONGODB_COLLECTION,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    relevance_score_fn="cosine",
)

#loader from html to text
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

assert len(docs) == 1

#splitter into equal chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)


#storing documents
document_ids = vector_store.add_documents(documents=all_splits)

print(document_ids[:3])

print(f"Split blog post into {len(all_splits)} sub-documents.")
print(f"Total characters: {len(docs[0].page_content)}")