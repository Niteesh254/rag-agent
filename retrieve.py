from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os
ChromaDB_path = './chroma_db'
Embedding_model = 'BAAI/bge-large-en-v1.5'
Top_k = 3
embeddings = HuggingFaceEmbeddings(
    model_name = Embedding_model,
    model_kwargs = {"device": "cpu"},
    encode_kwargs = {"normalize_embeddings": True}
)
vector_store = Chroma(  
    embedding_function=embeddings,
    persist_directory=ChromaDB_path
)

def retrieve(query):
    results = vector_store.similarity_search(query, k=Top_k)
    return results

if __name__ == "__main__":
    query ="Whats Niteesh's certifications?"
    results = retrieve(query)
    for i,doc in enumerate(results):
        print(f"Result {i+1}:")
        print(doc.page_content)
        print(f"Metadata: {doc.metadata}")
