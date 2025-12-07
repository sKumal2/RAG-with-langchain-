import getpass
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pymongo import MongoClient
import bs4

# from vector_store import vector_store
# from indexing import vector_store
from langchain.agents import create_agent
# from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
# from langsmith import uuid7

from langchain_core.tools import tool

load_dotenv()

model =ChatGoogleGenerativeAI(model = "gemini-2.5-flash-lite" )

MONGODB_URI = os.environ.get("MONGODB_URI")
ATLAS_VECTOR_SEARCH_INDEX_NAME = os.environ.get("ATLAS_VECTOR_SEARCH_INDEX_NAME")

#document embedding
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")


embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# Only keep post title, headers, and content from the full HTML.
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))

#mongodb atlas connection
client = MongoClient(MONGODB_URI)
MONGODB_COLLECTION = client["rag_db"]["rag_docs"]   # db_name.collection_name – change if you want

#vector store using mongodb
vector_store = MongoDBAtlasVectorSearch(    
    embedding=embeddings,
    collection=MONGODB_COLLECTION,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    relevance_score_fn="cosine",
    create_index =True,
)

#loader from html to text
loader = WebBaseLoader(
    # web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    web_paths=("https://docs.langchain.com/oss/python/langchain/rag/",),
    # bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

assert len(docs) == 1
print(f"Total characters: {len(docs[0].page_content)}")

"""
# Load and clean HTML from webpage
url = "https://docs.langchain.com/oss/python/langchain/rag/"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# Extract main content only
# Adjust the selector if the page uses different structure
main_content = soup.find("main")  # most docs wrap main content in <main>
if not main_content:
    main_content = soup  # fallback to full page if <main> not found

text = main_content.get_text(separator="\n", strip=True)
print(f"Total characters extracted: {len(text)}")

# Wrap cleaned text as a LangChain document
from langchain.schema import Document
docs = [Document(page_content=text, metadata={"source": url})]

"""

#splitter into equal chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # chunk size (characters)
    chunk_overlap=100,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)
print(f"Split into {len(all_splits)} chunks")

#storing documents
document_ids = vector_store.add_documents(documents=all_splits)
print(f"Indexed {len(document_ids)} documents")


# print(f"Split blog post into {len(all_splits)} sub-documents.")
# print(f"Total characters: {len(docs[0].page_content)}")