from typing import List, Dict, Any
from pathlib import Path
from config import Config
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd

class DocumentProcessor:
    """Handles PDF and Excel document loading and text splitting"""
    
    def __init__(self, chunk_size: int = Config.CHUNK_SIZE, 
                 chunk_overlap: int = Config.CHUNK_OVERLAP):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""]
        )
    
    def process_pdf(self, pdf_path: str) -> List[Document]:
        """Process a single PDF file"""
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata['source_file'] = pdf_path
                doc.metadata['file_name'] = Path(pdf_path).name
                doc.metadata['file_type'] = 'pdf'
            
            print(f"Loaded {len(documents)} pages from {Path(pdf_path).name}")
            return documents
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {e}")
            raise e
    
    def process_excel(self, excel_path: str) -> List[Document]:
        """Process an Excel file"""
        try:
            df = pd.read_excel(excel_path)
            documents = []
            
            # Convert each row to a document
            for idx, row in df.iterrows():
                content = " | ".join([f"{col}: {val}" for col, val in row.items()])
                doc = Document(
                    page_content=content,
                    metadata={
                        'source_file': excel_path,
                        'file_name': Path(excel_path).name,
                        'file_type': 'excel',
                        'row_index': idx
                    }
                )
                documents.append(doc)
            
            print(f"Loaded {len(documents)} rows from {Path(excel_path).name}")
            return documents
        except Exception as e:
            print(f"Error processing Excel {excel_path}: {e}")
            raise e
    
    def process_all_pdfs(self, dir_path: str) -> List[Document]:
        """Process all PDF files in a directory"""
        all_documents = []
        pdf_dir = Path(dir_path)
        pdf_files = list(pdf_dir.glob("**/*.pdf"))
        
        for pdf_file in pdf_files:
            documents = self.process_pdf(str(pdf_file))
            all_documents.extend(documents)
        
        print(f"Total PDF documents loaded: {len(all_documents)}")
        return all_documents
    
    def create_chunks(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        try:
            chunks = self.text_splitter.split_documents(documents)
            print(f"Created {len(chunks)} chunks from {len(documents)} documents")
            return chunks
        except Exception as e:
            print(f"Error during text splitting: {e}")
            raise e