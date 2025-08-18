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
    "https://your-app-name.streamlit.app",
    "https://your-app-name-staging.streamlit.app"  # If you have a staging app
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
    

@app.post("/rag")
def say_hello(payload: Query):
    """
    Endpoint that accepts a JSON input with a 'searchCriteria' key
    and returns a 'Hello [searchCriteria]!' message.
    """
    print(f" query: {payload}")
    return {"message": f"Hello {payload.query}!"}

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)

 