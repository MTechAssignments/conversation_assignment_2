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
from src.model.rag_system import RAGChunkIndexer, HybridRAGRetriever, GPT2AnswerGenerator, FinancialReportProcessor

zip_file_path = './data/gehc-annual-report-2023-2024.zip'
extracted_dir_path = './data/gehc_fin_extracted'
plain_text_dir_path = './data/gehc_fin_plain_text'

# Preprocess and build indexes (do this ONCE at startup)
processor = FinancialReportProcessor(zip_file_path, extracted_dir_path, plain_text_dir_path)
processor.extract_and_convert_html_to_text()
processor.load_plain_text_files()
processor.clean_all_texts()
processor.segment_reports()
processor.extract_financial_data()

indexer = RAGChunkIndexer(processor.cleaned_text_data)
indexer.create_chunks()
indexer.embed_chunks()
indexer.build_dense_index()
indexer.build_sparse_index()

retriever = HybridRAGRetriever(
    indexer.collection,
    indexer.embedding_model,
    indexer.tfidf_vectorizer,
    indexer.tfidf_matrix,
    indexer.embedded_chunks
)
retriever.load_cross_encoder()
answer_generator = GPT2AnswerGenerator()

def invoke_rag(user_query, max_length=512):
    response = retriever.get_user_response(user_query, answer_generator)
    data = {
        "Question": user_query,
        "Answer": response["answer"],
        "Chunks": response["chunks"]
        }
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

