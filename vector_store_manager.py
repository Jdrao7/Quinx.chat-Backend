from typing import List, Dict, Any
from config import Config
import chromadb
import os
import uuid
from langchain_core.documents import Document
import numpy as np


class VectorStoreManager:
    """Manages ChromaDB vector store"""
    
    def __init__(self, collection_name: str = Config.COLLECTION_NAME, 
                 persist_directory: str = Config.VECTOR_STORE_PATH):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize ChromaDB vector store"""
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"similarity_search": "cosine"}
            )
            print(f"ChromaDB Vector Store initialized at {self.persist_directory}")
            print(f"Existing documents in collection '{self.collection_name}': {self.collection.count()}")
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            raise e
    
    def add_documents(self, documents: List[Document], embeddings: np.ndarray):
        """Add documents with embeddings to the vector store"""
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents and embeddings must match")
        
        ids = []
        metadatas = []
        documents_texts = []
        embeddings_list = []
        
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)
            
            documents_texts.append(doc.page_content)
            embeddings_list.append(embedding.tolist())
        
        try:
            self.collection.add(
                ids=ids,
                metadatas=metadatas,
                documents=documents_texts,
                embeddings=embeddings_list
            )
            print(f"Added {len(documents_texts)} documents to vector store. Total: {self.collection.count()}")
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise e
    
    def query(self, query_embeddings: np.ndarray, top_k: int = Config.TOP_K_RESULTS) -> Dict[str, Any]:
        """Query the vector store"""
        try:
            results = self.collection.query(
                query_embeddings=query_embeddings.tolist(),
                n_results=top_k
            )
            return results
        except Exception as e:
            print(f"Error querying vector store: {e}")
            raise e
    
    def reset_collection(self):
        """Delete and recreate the collection"""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"similarity_search": "cosine"}
            )
            print(f"Collection '{self.collection_name}' reset successfully")
        except Exception as e:
            print(f"Error resetting collection: {e}")
            raise e