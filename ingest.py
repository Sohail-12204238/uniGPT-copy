import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_qdrant import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# --- Configuration ---
PDF_DIR = "D:/uniGPT_project/documents"
VECTOR_STORE_PATH = "./vector_store"
COLLECTION_NAME = "Uni_data"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

def ingest_data():
    """
    Loads PDFs, splits them into chunks, creates embeddings,
    and stores them in a Qdrant vector database.
    """
    # 1. Load PDF Documents
    all_docs = []
    print(f"Loading PDFs from {PDF_DIR}...")
    for filename in os.listdir(PDF_DIR):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(PDF_DIR, filename))
            docs = loader.load()
            all_docs.extend(docs)
    print(f"Loaded {len(all_docs)} pages from PDFs.")

    # 2. Split Documents into Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(all_docs)
    print(f"Split documents into {len(chunks)} chunks.")

    # 3. Setup Embeddings Model
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME, encode_kwargs=encode_kwargs)

    # 4. Setup Qdrant Client and Collection
    # This will create a local, file-based Qdrant database
    client = QdrantClient(path=VECTOR_STORE_PATH)

    # Check if collection already exists, if not create it
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]

    if COLLECTION_NAME not in collection_names:
        print(f"Creating new Qdrant collection: {COLLECTION_NAME}")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
    else:
        print(f"Using existing collection: {COLLECTION_NAME}")


    # 5. Add Chunks to Qdrant Vector Store
    print("Adding document chunks to the vector store...")
    Qdrant.from_documents(
        documents=chunks,
        embedding=embeddings,
        path=VECTOR_STORE_PATH,
        collection_name=COLLECTION_NAME,
    )
    print("--- Ingestion Complete! ---")

if __name__ == "__main__":
    ingest_data()