#Take the query and chunks as input
# Build the RAG prompt
# Call the Ollama LLM with the RAG prompt
#Return the answer to the user
import httpx
import json
from retrieve import retrieve
Ollama_BASE_URL = "http://localhost:11434"
Ollama_MODEL = "llama3.1:8b"
def build_rag_prompt(query,chunks):
    context = ""
    for i,chunk in enumerate(chunks):
        context += f"Chunk {i+1}:\n{chunk.page_content}\n\n"
    prompt = f"""You are a helpful assistant. Use the following retrieved chunks to answer the query. If the answer is not present in the chunks, say "I cannot find this Information"
    Question: {query}
    Context: {context}
    Answer:"""
    return prompt
def generate_answer(query,chunks):
    prompt = build_rag_prompt(query,chunks)
    response = httpx.post(
        f"{Ollama_BASE_URL}/api/generate",
        json={
            "model": Ollama_MODEL,
            "prompt": prompt,
            "stream": False,
        },        timeout=120

    )
    if response.status_code == 200:
        data = response.json()
        return data["response"].strip()
    else:
        print(f"Error: {response.status_code}")
        return None
if __name__ == "__main__":    
    
    
    
    
    query = "List all the skills mentioned in the documents."
    chunks = retrieve(query)
    
    answer = generate_answer(query, chunks)
    print("\nAnswer:")
    print(answer)
