from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
Docs_path = './documents'
Chunk_size = 512
Chunk_overlap = 50
ChromaDB_path = './chroma_db'
Embedding_model = 'BAAI/bge-large-en-v1.5'
#Step 1: Load all the pdfs from the Documents folder
def load_documents():
    documents = []
    pdf_files = [f for f in os.listdir(Docs_path) if f.endswith('.pdf')]
    for pdf in pdf_files:
        pdf_path = os.path.join(Docs_path,pdf)
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
    return documents

#Step 2: Split the text into chunks
def chunk_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Chunk_size,
        chunk_overlap=Chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

#Step 3: Load the BAAI embedding model
# Step 4: Convert all the chunks into vectors and store in chromaDB
def create_vector_store(chunks):
    embedding_model = HuggingFaceEmbeddings(
        model_name = Embedding_model,
        model_kwargs = {"device": "cpu"},
        encode_kwargs = {"normalize_embeddings": True}
    )
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=ChromaDB_path
    )
    return vector_store
def main():
    print("Starting the ingestion process...")
    documents = load_documents()
    print(f"Documents loaded:{len(documents)}")
    chunks = chunk_documents(documents)
    print(f"Chunks created:{len(chunks)}")
    vector_store = create_vector_store(chunks)
    print("Ingestion process completed successfully.")
if __name__ == "__main__":
    main()