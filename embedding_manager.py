from typing import List, Dict, Any
from config import Config
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingManager:
    """Manages sentence transformer embeddings"""
    
    def __init__(self, model_name: str = Config.EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            self.model = SentenceTransformer(self.model_name)
            print(f"Model {self.model_name} loaded successfully with dimensions: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise e
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Generated embeddings for {len(texts)} chunks")
        return embeddings