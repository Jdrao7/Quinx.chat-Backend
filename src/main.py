from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import os
import shutil
from rag_system import RAGSystem 
from config import Config  

app = FastAPI(title="RAG System API with PDF/Excel Support", version="2.0.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG instance
rag_system = RAGSystem()

# Create upload directory
os.makedirs(Config.UPLOAD_DIR, exist_ok=True)

# Pydantic models for API
class QueryInput(BaseModel):
    question: str
    top_k: Optional[int] = Config.TOP_K_RESULTS

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

class StatusResponse(BaseModel):
    status: str
    message: str
    details: Optional[Dict[str, Any]] = None

# ============= API Endpoints =============
@app.post("/upload-pdf", response_model=StatusResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and ingest a PDF file"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Save uploaded file
        file_path = os.path.join(Config.UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Ingest the PDF
        rag_system.ingest_pdf(file_path)
        
        return StatusResponse(
            status="success",
            message=f"PDF '{file.filename}' uploaded and ingested successfully",
            details={"file_path": file_path}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-excel", response_model=StatusResponse)
async def upload_excel(file: UploadFile = File(...)):
    """Upload and ingest an Excel file"""
    if not (file.filename.endswith('.xlsx') or file.filename.endswith('.xls')):
        raise HTTPException(status_code=400, detail="Only Excel files (.xlsx, .xls) are allowed")
    
    try:
        # Save uploaded file
        file_path = os.path.join(Config.UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Ingest the Excel file
        rag_system.ingest_excel(file_path)
        
        return StatusResponse(
            status="success",
            message=f"Excel file '{file.filename}' uploaded and ingested successfully",
            details={"file_path": file_path}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-multiple")
async def upload_multiple_files(files: List[UploadFile] = File(...)):
    """Upload and ingest multiple PDF/Excel files"""
    results = []
    
    for file in files:
        try:
            # Save uploaded file
            file_path = os.path.join(Config.UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Ingest based on file type
            if file.filename.endswith('.pdf'):
                rag_system.ingest_pdf(file_path)
            elif file.filename.endswith(('.xlsx', '.xls')):
                rag_system.ingest_excel(file_path)
            else:
                results.append({
                    "filename": file.filename,
                    "status": "skipped",
                    "message": "Unsupported file type"
                })
                continue
            
            results.append({
                "filename": file.filename,
                "status": "success",
                "message": "Ingested successfully"
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": str(e)
            })
    
    return {"results": results}

@app.post("/query", response_model=QueryResponse)
async def query_documents(query: QueryInput):
    """Query the RAG system"""
    try:
        result = rag_system.query(query.question, query.top_k)
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get statistics about the vector store"""
    try:
        count = rag_system.vector_store.collection.count()
        return {
            "total_documents": count,
            "collection_name": Config.COLLECTION_NAME,
            "embedding_model": Config.EMBEDDING_MODEL,
            "llm_model": Config.GROQ_MODEL
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/reset")
async def reset_database():
    """Reset the entire vector database"""
    try:
        rag_system.vector_store.reset_collection()
        return {"status": "success", "message": "Vector database reset successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "2.0.0"}


if __name__ == "__main__":
    print("Starting RAG System with PDF/Excel Support...")
    print(f"Vector store path: {Config.VECTOR_STORE_PATH}")
    print(f"Upload directory: {Config.UPLOAD_DIR}")
    print("\nStarting FastAPI server on http://localhost:8000")
    print("API docs available at http://localhost:8000/docs")
    print("\nEndpoints:")
    print("  POST /upload-pdf - Upload a PDF file")
    print("  POST /upload-excel - Upload an Excel file")
    print("  POST /upload-multiple - Upload multiple files")
    print("  POST /query - Query the documents")
    print("  GET  /stats - Get database statistics")
    print("  DELETE /reset - Reset the database")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)