from langchain_community.document_loaders import PyPDFLoader
from  pathlib import Path
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
pdf_path=Path("__file__").parent/"Hallucination_Firewall_Business_Plan.pdf"
loader=PyPDFLoader(file_path=pdf_path)
docs=loader.load()
text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
split_docs=text_splitter.split_documents(
    docs
)
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
)
embeddings.embed_query("What's our Q1 revenue?")