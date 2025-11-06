from typing import List, Dict, Any
from config import Config
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingManager:
    """Manages sentence transformer embeddings with lazy loading"""
    
    def __init__(self, model_name: str = Config.EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = None # Initialize model as None
        # DO NOT load the model here.
        # self._load_model() # <--- REMOVED from __init__
    
    def _load_model(self):
        """Internal method to load the sentence transformer model"""
        print(f"Loading model {self.model_name} for the first time...")
        try:
            self.model = SentenceTransformer(self.model_name)
            print(f"Model {self.model_name} loaded successfully with dimensions: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            self.model = None # Ensure model is None if loading fails
            raise e
            
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        # --- LAZY LOADING CHECK ---
        # Load the model only if it hasn't been loaded yet.
        if self.model is None:
            self._load_model()
        # --- END LAZY LOADING CHECK ---
            
        if self.model is None:
            raise ValueError("Model could not be loaded")
        
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print("Embeddings generated.")
        return embeddings