import os
from dotenv import load_dotenv
load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    GROQ_MODEL = "llama-3.1-8b-instant"  # or "mixtral-8x7b-32768"
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100
    TOP_K_RESULTS = 3
    VECTOR_STORE_PATH = "./data/vec_store"
    UPLOAD_DIR = "./uploads"
    COLLECTION_NAME = "pdf_documents"