from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import BaseModel
import uvicorn

# Create a FastAPI instance
app = FastAPI(
    title="RAG API Server",
    version="1.0",
    docs_url="/docs",
    description="RAG API Server"
)

allowed_origins = [
    "http://localhost:8000",
    "https://conv-ai-assignment-2-grp-16-rag-finetune.streamlit.app",
    "https://conversation-assignment-2.onrender.com",
    "https://conversation-assignment-2.onrender.com:8000"
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define a Pydantic model to handle JSON input
class Query(BaseModel):
    query: str
    max_length: int
    


# --- RAG System Initialization (runs once at server startup) ---
from src.api.rag_system import FinancialQASystem
# Initialize and setup
qa_system = FinancialQASystem()
qa_system.setup_data()
qa_system.setup_rag_system()


# RAG pipeline objects are initialized ONCE above, at server startup.
def invoke_rag(user_query, max_length=512):
    # Only use already-initialized retriever and answer_generator
    response = qa_system.get_response(user_query)
    # Only include the answer and metadata
    data = {
        "Question": user_query,  # Add the question from the input parameter
        "Method": response["Method"],
        "Answer": response["Answer"],
        "Confidence": response["Confidence"],
        "Time (s)": response["Time (s)"]
    }
    if response.get("Chunks"):
        data["ContextSnippet"] = response["Chunks"][0]["content"][:100]
    return data

# --- API Endpoint ---
@app.post("/rag")
def rag_query(payload: Query):
    """
    Accepts a user query and returns the RAG-generated answer and supporting chunks.
    """
    user_query = payload.query
    data = invoke_rag(payload.query, payload.max_length)
    
    return data

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "message": "API Server is healthy!"}
    

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)

