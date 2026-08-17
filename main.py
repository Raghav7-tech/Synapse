import os
from pathlib import Path
from dotenv import load_dotenv 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import FastEmbedEmbeddings
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") 
pdf_path = Path(__file__).parent / "manan- a techno surge.pdf"
loader = PyPDFLoader(file_path=str(pdf_path))
docs = loader.load() 
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200)
split_docs = text_splitter.split_documents(docs) 
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vector_store = QdrantVectorStore.from_documents(
    documents=split_docs,
    collection_name="pdf-synapse",
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    embedding=embeddings,
    force_recreate=True
) 
print("pdf analysis done !!")