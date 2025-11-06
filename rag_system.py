from typing import List, Dict, Any, Optional
from config import Config
from document_processor import DocumentProcessor
from embedding_manager import EmbeddingManager
from vector_store_manager import VectorStoreManager
from rag_retriever import RAGRetriever

class RAGSystem:
    """Main RAG system orchestrating all components"""
    
    def __init__(self, groq_api_key: str = Config.GROQ_API_KEY):
        self.doc_processor = DocumentProcessor()
        self.embedding_manager = EmbeddingManager()
        self.vector_store = VectorStoreManager()
        self.retriever = RAGRetriever(
            self.vector_store, 
            self.embedding_manager,
            groq_api_key
        )
    
    def ingest_pdf(self, pdf_path: str):
        """Ingest a single PDF file"""
        # Load PDF
        documents = self.doc_processor.process_pdf(pdf_path)
        
        # Create chunks
        chunks = self.doc_processor.create_chunks(documents)
        
        # Generate embeddings
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embedding_manager.generate_embeddings(texts)
        
        # Add to vector store
        self.vector_store.add_documents(chunks, embeddings)
    
    def ingest_excel(self, excel_path: str):
        """Ingest an Excel file"""
        # Load Excel
        documents = self.doc_processor.process_excel(excel_path)
        
        # Create chunks
        chunks = self.doc_processor.create_chunks(documents)
        
        # Generate embeddings
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embedding_manager.generate_embeddings(texts)
        
        # Add to vector store
        self.vector_store.add_documents(chunks, embeddings)
    
    def ingest_directory(self, dir_path: str):
        """Ingest all PDFs from a directory"""
        # Load all PDFs
        documents = self.doc_processor.process_all_pdfs(dir_path)
        
        # Create chunks
        chunks = self.doc_processor.create_chunks(documents)
        
        # Generate embeddings
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embedding_manager.generate_embeddings(texts)
        
        # Add to vector store
        self.vector_store.add_documents(chunks, embeddings)
    
    def query(self, question: str, top_k: int = Config.TOP_K_RESULTS) -> Dict[str, Any]:
        """Query the RAG system"""
        # Retrieve relevant documents
        results = self.retriever.retrieve(question, top_k)
        
        if not results['documents'] or not results['documents'][0]:
            return {
                "answer": "No relevant documents found in the database.",
                "sources": []
            }
        
        # Prepare context from retrieved documents
        context = "\n\n".join(results['documents'][0])
        
        # Generate answer
        answer = self.retriever.generate_answer(question, context)
        
        # Prepare response
        sources = []
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            sources.append({
                "content": doc,
                "metadata": metadata,
                "relevance_rank": i + 1
            })
        
        return {
            "answer": answer,
            "sources": sources
        }
