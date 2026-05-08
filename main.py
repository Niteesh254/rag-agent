from fastapi import FastAPI, Request
from pydantic import BaseModel
from generate import generate_answer
from retrieve import retrieve
#Server need to take post request from client
app = FastAPI()
class ChatRequest(BaseModel):
    query: str
@app.post("/chat")
# Call the retrieve function to get the relevant chunks
# Call the generate function with the query and retrieved chunks to get the answer
# Return the answer to the client
def chat(request: ChatRequest):
    query = request.query
    answer = generate_answer(query, retrieve(query))
    return {"answer": answer}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)