from typing import List, Dict, Any, Optional
from config import Config
from document_processor import DocumentProcessor
from embedding_manager import EmbeddingManager
from vector_store_manager import VectorStoreManager
from langchain_groq import ChatGroq

class RAGRetriever:
    """Handles retrieval and generation"""
    
    def __init__(self, vector_store: VectorStoreManager, 
                 embedding_manager: EmbeddingManager,
                 groq_api_key: str = Config.GROQ_API_KEY):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=Config.GROQ_MODEL,
            temperature=0.7
        )
    
    def retrieve(self, query: str, top_k: int = Config.TOP_K_RESULTS) -> Dict[str, Any]:
        """Retrieve relevant documents"""
        query_embeddings = self.embedding_manager.generate_embeddings([query])
        results = self.vector_store.query(query_embeddings, top_k)
        return results
    
    def generate_answer(self, query: str, context: str) -> str:
        """Generate answer using Groq LLM"""
        prompt = f"""Use the following context to answer the question. If you don't know the answer based on the context, say so.

Context: {context}

Question: {query}

Answer:"""
        
        response = self.llm.invoke(prompt)
        return response.content
