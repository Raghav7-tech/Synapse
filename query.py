import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate 
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

print("Loading embedding model...")
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

print("Connecting to Qdrant Cloud with extended timeout...")
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60,
)
vector_store = QdrantVectorStore(
    client=client,
    collection_name="pdf-synapse",
    embedding=embeddings,
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash", 
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3
)

system_prompt = (
    "You are a helpful assistant. Use the following retrieved context "
    "to answer the user's question. If you don't know the answer based "
    "on the context, try  to  give  the answer using  your  capabilities.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

question = input("enter  your  question  here -  ")

print(f"\nAsking Gemini: '{question}'...\n")
response = rag_chain.invoke({"input": question})

print("--- GEMINI'S ANSWER ---")
print(response["answer"])