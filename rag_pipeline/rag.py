import openai
from documents import *

class BasicRAG:
    """
    A default RAG (Retrieval-Augmented Generation) pipeline that retrieves relevant documents 
    from ChromaDB based on a user query and generates a response using OpenAI's API.
    """
    
    def __init__(self, document_store: DocumentStore, model: str = "gpt-4"):
        self.document_store = document_store
        self.model = model
    
    def retrieve_documents(self, query: str, top_k: int = 5):
        """Fetches the top-k most relevant documents from the document store."""
        return self.document_store.query_documents(query_text=query, top_k=top_k)
    
    def generate_response(self, query: str, retrieved_docs: list) -> str:
        """Generates a response using OpenAI's GPT model with retrieved documents as context."""
        context = "\n\n".join([doc["metadata"]["title"] + ": " + doc["metadata"].get("objectives", "") for doc in retrieved_docs])
        prompt = f"""
        You are an AI assistant helping users learn from Adobe's digital experience courses.
        Based on the following retrieved documents, answer the query:
        
        Retrieved Documents:
        {context}
        
        User Query: {query}
        """
        
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        return response["choices"][0]["message"]["content"]
    
    def run_rag_pipeline(self, query: str, top_k: int = 5) -> str:
        """Runs the full RAG pipeline: retrieves documents and generates a response."""
        retrieved_docs = self.retrieve_documents(query, top_k)
        response = self.generate_response(query, retrieved_docs)
        return response