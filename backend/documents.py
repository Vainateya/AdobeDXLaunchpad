import chromadb
import openai
import numpy as np
from typing import Dict, List, Union
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dependency_graph')))
from utils import *

class DocumentStore:
    def __init__(self, embedding_model: str = "text-embedding-ada-002", similarity_metric: str = "cosine", storage_path: str = ".chroma_db"):
        self.storage_path = storage_path  # Path where ChromaDB will store data
        self.client = chromadb.PersistentClient(path=self.storage_path)
        self.similarity_metric = similarity_metric  # Store the similarity setting
        self.collection = self.client.get_or_create_collection(
            name="learning_documents",
            metadata={"hnsw:space": self._get_similarity_function()}  # Apply metric setting
        )
        self.embedding_model = embedding_model
    
    def _get_similarity_function(self) -> str:
        """Maps user-selected metric to ChromaDB's supported metrics."""
        metric_mapping = {
            "cosine": "cosine",
            "euclidean": "l2",  # ChromaDB calls Euclidean "l2"
            "dot_product": "ip"  # ChromaDB calls Dot Product "ip" (Inner Product)
        }
        return metric_mapping.get(self.similarity_metric, "cosine")  # Default to cosine
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generates an embedding for a given text using OpenAI's embedding model."""
        response = openai.embeddings.create(
            model=self.embedding_model, 
            input=[text], 
        )
        return response.data[0].embedding
    
    def process_source(self, source: Union['Course', 'Certificate', 'TextDocument']) -> Dict:
        """Extracts relevant text and metadata from Course or Certificate objects."""
        if isinstance(source, Course):
            text = f"{source.objectives} " + " ".join([m.description for m in source.modules])
            metadata = {
                "type": "course",
                "title": source.display,
                "category": source.category,
                "level": source.level,
                "job_role": source.job_role,
                "objectives": source.objectives,
                "modules": ", ".join([m.title for m in source.modules]) 
            }
        elif isinstance(source, Certificate):
            text = f"{source.prereq} " + " ".join(source.study_materials)
            metadata = {
                "type": "certificate",
                "title": source.display,
                "category": source.category,
                "level": source.level,
                "job_role": source.job_role,
                "prereq": source.prereq,
                "study_materials": "; ".join(source.study_materials)
            }
        elif isinstance(source, TextDocument):
            text = source.content
            metadata = {
                "type": "program_info",
                "title": source.display,
                "filename": os.path.basename(source.filepath)
            }
        else:
            raise ValueError("Unsupported document type.")
        
        embedding = self.generate_embedding(text)
        return {"text": text, "embedding": embedding, "metadata": metadata}
    
    def add_document(self, source: Union['Course', 'Certificate']):
        """Adds a processed document to ChromaDB."""
        document = self.process_source(source)
        self.collection.add(
            ids=[source.display],
            embeddings=[document["embedding"]],
            metadatas=[document["metadata"]]
        )
    
    def query_documents(self, query_text: str, top_k: int = 1) -> List[Dict]:
        """Retrieves the top K most relevant documents based on similarity search."""
        query_embedding = self.generate_embedding(query_text)
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        
        retrieved_documents = []
        for i in range(len(results["ids"][0])):
            retrieved_documents.append({
                "id": results["ids"][0][i],
                "metadata": results["metadatas"][0][i]
            })
        return retrieved_documents

    def get_category_from_best_document(self, query_text):
        query_embedding = self.generate_embedding(query_text)
        results = self.collection.query(query_embeddings=[query_embedding], n_results=1)
        
        return results['metadatas'][0][0]['category']
    
    def delete_document(self, document_id: str):
        """Deletes a document from ChromaDB by ID."""
        self.collection.delete(ids=[document_id])


