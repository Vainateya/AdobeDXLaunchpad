#!/usr/bin/env python3
"""
Simple command-line version of the Adobe Resources RAG chatbot with agentic capabilities.
This version doesn't require Streamlit and is useful for testing.
"""

import json
import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from sentence_transformers import SentenceTransformer
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Load environment variables
load_dotenv()

class ChatMessage:
    """Represents a chat message with metadata."""
    def __init__(self, role: str, content: str, timestamp: Optional[datetime] = None, metadata: Optional[Dict] = None):
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict:
        return {
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ChatMessage':
        return cls(
            role=data['role'],
            content=data['content'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            metadata=data.get('metadata', {})
        )

class SimpleAgenticRAG:
    """Enhanced RAG system with agentic capabilities and chat history."""
    
    def __init__(self):
        """Initialize the simple agentic RAG system."""
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize sentence transformer (optional)
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"Warning: Could not load SentenceTransformer: {e}")
            self.model = None
        
        # Initialize ChromaDB with new API
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Load and process resources
        self.resources = self.load_resources()
        self.setup_vector_store()
        
        # Chat history
        self.chat_history: List[ChatMessage] = []
        self.user_profile = None  # Store user profile from survey

    def set_user_profile(self, profile_dict: dict):
        """Set the user profile from survey data."""
        self.user_profile = profile_dict

    def get_user_profile(self):
        return self.user_profile

    def load_resources(self) -> Dict[str, Any]:
        """Load resources from JSON file."""
        with open('resources.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def create_documents(self) -> List[Dict[str, Any]]:
        """Convert resources into documents for vector storage."""
        documents = []
        
        for title, data in self.resources.items():
            content = f"""
            Title: {title}
            Type: {data.get('type', 'N/A')}
            Category: {data.get('category', 'N/A')}
            Level: {data.get('level', 'N/A')}
            Job Role: {data.get('job_role', 'N/A')}
            Objectives: {data.get('objectives', 'N/A')}
            Modules: {data.get('modules', 'N/A')}
            Prerequisites: {data.get('prereq', 'N/A')}
            Study Materials: {data.get('study_materials', 'N/A')}
            Exam Details: {data.get('details', 'N/A')}
            """
            
            documents.append({
                'id': title,
                'content': content,
                'metadata': {
                    'title': title,
                    'type': data.get('type', 'N/A'),
                    'category': data.get('category', 'N/A'),
                    'level': data.get('level', 'N/A'),
                    'job_role': data.get('job_role', 'N/A')
                }
            })
        
        return documents
    
    def setup_vector_store(self):
        """Set up the vector store with documents."""
        try:
            self.collection = self.chroma_client.get_collection("adobe_resources")
            print("Loaded existing vector database")
        except:
            print("Creating new vector database...")
            self.collection = self.chroma_client.create_collection("adobe_resources")
            
            documents = self.create_documents()
            
            for doc in documents:
                self.collection.add(
                    documents=[doc['content']],
                    metadatas=[doc['metadata']],
                    ids=[doc['id']]
                )
            
            print("Vector database created successfully!")
    
    def decompose_query(self, query: str) -> List[str]:
        """Decompose complex queries into simpler sub-queries."""
        system_prompt = """You are an expert at breaking down complex questions about Adobe resources into simpler, more focused sub-questions.

Given a complex question, break it down into 2-4 simpler questions that can be answered individually and then combined to provide a comprehensive answer.

Return only the sub-questions as a JSON array of strings. Do not include any explanations or additional text.

Example:
Input: "What Adobe Analytics courses are available for beginners and what are their prerequisites?"
Output: ["What Adobe Analytics courses are available?", "What are the prerequisites for Adobe Analytics courses?", "Which Adobe Analytics courses are suitable for beginners?"]"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Break down this question: {query}"}
                ],
                max_tokens=200,
                temperature=0.3
            )
            
            # Parse JSON response
            content = response.choices[0].message.content
            print(f"decompose_query content: {content}")
            if content and content.strip().startswith('[') and content.strip().endswith(']'):
                return json.loads(content.strip())
            else:
                # Fallback: return original query
                return [query]
        except:
            return [query]
    
    def search_with_reasoning(self, query: str, n_results: int = 5) -> List[Dict]:
        """Enhanced search with reasoning about what to look for."""
        system_prompt = """You are an expert at understanding what information is needed to answer questions about Adobe resources.

Given a question, identify the key concepts, categories, levels, and job roles that would be most relevant to finding the answer.

Return a JSON object with:
- "search_terms": List of key terms to search for
- "categories": List of relevant Adobe product categories
- "levels": List of relevant certification levels
- "job_roles": List of relevant job roles
- "reasoning": Brief explanation of why these are relevant"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze this question: {query}"}
                ],
                max_tokens=300,
                temperature=0.3
            )
            
            content = response.choices[0].message.content
            if content:
                analysis = json.loads(content.strip())
                
                # Use the analysis to enhance search
                enhanced_query = f"{query} {' '.join(analysis.get('search_terms', []))}"
                
                # Search with enhanced query
                results = self.collection.query(
                    query_texts=[enhanced_query],
                    n_results=n_results
                )
                
                documents = []
                if results and results.get('documents') and results['documents'][0]:
                    for i in range(len(results['documents'][0])):
                        documents.append({
                            'content': results['documents'][0][i],
                            'metadata': results['metadatas'][0][i] if results.get('metadatas') else {},
                            'distance': results['distances'][0][i] if results.get('distances') else None,
                            'reasoning': analysis.get('reasoning', '')
                        })
                
                return documents
        except:
            pass
        
        # Fallback to simple search
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        documents = []
        if results and results.get('documents') and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                documents.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results.get('metadatas') else {},
                    'distance': results['distances'][0][i] if results.get('distances') else None
                })
        
        return documents
    
    def get_chat_context(self, max_messages: int = 5) -> str:
        """Get recent chat context for conversation continuity."""
        if not self.chat_history:
            return ""
        
        recent_messages = self.chat_history[-max_messages:]
        context = "Recent conversation context:\n"
        
        for msg in recent_messages:
            role = "User" if msg.role == "user" else "Assistant"
            context += f"{role}: {msg.content}\n"
        
        return context
    
    def create_system_prompt(self) -> str:
        """Create the system prompt for the chatbot."""
        return """You are an expert assistant specializing in Adobe certification courses and resources. You have comprehensive knowledge of Adobe's learning programs, certifications, and training materials.

Your expertise includes:
- Adobe Experience Cloud products (Analytics, Target, Campaign, etc.)
- Adobe Creative Cloud products
- Adobe Document Cloud
- Adobe Commerce (formerly Magento)
- Adobe Experience Manager
- Adobe Workfront
- Adobe Marketo Engage
- Adobe Journey Optimizer
- Adobe Real-Time CDP
- And many other Adobe products

You can help users with:
1. Finding appropriate courses and certifications based on their experience level and job role
2. Understanding course objectives, modules, and prerequisites
3. Providing detailed information about exam requirements and study materials
4. Recommending learning paths for different career goals
5. Explaining Adobe product features and capabilities
6. Answering questions about certification costs, exam formats, and passing scores

IMPORTANT: Always reference the conversation history when relevant. If the user refers to previous questions or asks follow-up questions, use the context from earlier in the conversation to provide more relevant and contextual answers.

If the user asks about your capabilities, limitations, or how you work, explain that you are an AI-powered assistant for Adobe learning and certification. Describe your main features (course/cert search, learning path graph, survey for personalization), what you’re best at, and what you cannot do (e.g., register for exams, provide official support, answer non-Adobe questions). Be clear, concise, and helpful.

Always provide accurate, helpful information based on the available resources. If you're unsure about something, say so rather than making up information."""
    
    def create_context_prompt(self, query: str, relevant_docs: List[Dict], chat_context: str = "") -> str:
        """Create a context prompt with relevant documents, user profile, and chat history."""
        context = "Based on the following Adobe resources information, user profile, and conversation context, please answer the user's question:\n\n"
        if self.user_profile:
            context += f"User profile: {json.dumps(self.user_profile, indent=2)}\n\n"
        if chat_context:
            context += f"{chat_context}\n\n"
        for i, doc in enumerate(relevant_docs, 1):
            context += f"Resource {i}:\n{doc['content']}\n\n"
        context += f"User Question: {query}\n\n"
        context += "Please provide a comprehensive answer based on the information above. If the information isn't available in the provided resources, please say so. If this is a follow-up question, reference the conversation context appropriately. Please answer in an HTML body, beginning with <html> and ending with </html>."
        return context
    
    def get_response(self, query: str) -> str:
        """Get response from the agentic RAG system."""
        try:
            # Get chat context
            start_time = time.time()
            chat_context = self.get_chat_context()
            print(f"A: {time.time() - start_time}")
            
            # Decompose complex queries
            start_time = time.time()
            sub_queries = self.decompose_query(query)
            print(f"B: {time.time() - start_time}")
            print(f"sub_queries: {sub_queries}")
            if not sub_queries or not isinstance(sub_queries, list):
                print("sub_queries is empty or not a list! Falling back to original query.")
                sub_queries = [query]
            
            # Search for relevant documents with reasoning in parallel
            start_time = time.time()
            all_relevant_docs = []
            with ThreadPoolExecutor(max_workers=min(len(sub_queries), 5)) as executor:
                # Submit all search tasks
                future_to_query = {
                    executor.submit(self.search_with_reasoning, sub_query, 3): sub_query 
                    for sub_query in sub_queries
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_query):
                    try:
                        docs = future.result()
                        all_relevant_docs.extend(docs)
                    except Exception as e:
                        print(f"Warning: Search failed for a sub-query: {e}")
                        continue
            print(f"C: {time.time() - start_time}")

            # Remove duplicates based on content
            start_time = time.time()
            seen_contents = set()
            unique_docs = []
            for doc in all_relevant_docs:
                content_hash = hash(doc['content'][:100])  # Use first 100 chars as hash
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    unique_docs.append(doc)
            
            # Limit to top 5 most relevant
            unique_docs = unique_docs[:5]
            print(f"D: {time.time() - start_time}")

            # Create context prompt
            start_time = time.time()
            context_prompt = self.create_context_prompt(query, unique_docs, chat_context)
            
            # Get response from OpenAI
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.create_system_prompt()},
                    {"role": "user", "content": context_prompt}
                ],
                max_tokens=800,
                stream=True,
                temperature=0.7
            )
            print(f"E: {time.time() - start_time}")

            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_response += token
                    if token not in ["html", "```"]:
                        yield token
            
            # Extract courses/certificates from the full response
            extracted_items = self.extract_courses_certificates(full_response)
            if extracted_items:
                print(f"Extracted courses/certificates: {extracted_items}")
                # Store for later use in graph generation
                self.last_extracted_items = extracted_items
            else:
                print(f"no Extracted courses/certificates :(")
                self.last_extracted_items = []
            
        except Exception as e:
            print(f"Exception in get_response after B: {e}")
            return f"Error: {str(e)}"
    
    def get_last_extracted_items(self) -> List[str]:
        """Get the last extracted course/certificate items."""
        return getattr(self, 'last_extracted_items', [])
    
    def add_message(self, role: str, content: str, metadata: Dict = None):
        """Add a message to chat history."""
        message = ChatMessage(role=role, content=content, metadata=metadata or {})
        self.chat_history.append(message)
    
    def get_chat_summary(self) -> str:
        """Get a summary of the conversation for context."""
        if len(self.chat_history) < 3:
            return ""
        
        try:
            recent_messages = self.chat_history[-10:]  # Last 10 messages
            conversation_text = "\n".join([f"{msg.role}: {msg.content}" for msg in recent_messages])
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Summarize the key points from this conversation in 2-3 sentences."},
                    {"role": "user", "content": conversation_text}
                ],
                max_tokens=150,
                temperature=0.3
            )
            
            return response.choices[0].message.content
        except:
            return ""
    
    def extract_courses_certificates(self, response_text: str) -> List[str]:
        """Extract course and certificate names from the RAG response."""
        try:
            # Use OpenAI to extract course/certificate names from the response
            extraction_prompt = f"""
            Extract Adobe course and certificate names from the following response text.
            Return only the names as a JSON array of strings. If no courses or certificates are mentioned, return an empty array.
            
            Response text:
            {response_text}
            
            Return format: ["Course Name 1", "Certificate Name 2", ...]
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at extracting Adobe course and certificate names from text. Return only valid Adobe course/certificate names as a JSON array."},
                    {"role": "user", "content": extraction_prompt}
                ],
                max_tokens=200,
                temperature=0.1
            )
            
            content = response.choices[0].message.content.replace("`", "").replace("json", "")
            print(f"Extracted courses/certificates: {content}")
            if content and content.strip().startswith('[') and content.strip().endswith(']'):
                extracted_names = json.loads(content.strip())
                # Filter out any non-string items and clean up
                cleaned_names = [name.strip() for name in extracted_names if isinstance(name, str) and name.strip()]
                return cleaned_names
            else:
                return []
        except Exception as e:
            print(f"Error extracting courses/certificates: {e}")
            return []
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get a summary of available resources."""
        summary = {
            'total_resources': len(self.resources),
            'by_type': {},
            'by_category': {},
            'by_level': {},
            'by_job_role': {}
        }
        
        for title, data in self.resources.items():
            resource_type = data.get('type', 'Unknown')
            summary['by_type'][resource_type] = summary['by_type'].get(resource_type, 0) + 1
            
            category = data.get('category', 'Unknown')
            summary['by_category'][category] = summary['by_category'].get(category, 0) + 1
            
            level = data.get('level', 'Unknown')
            summary['by_level'][level] = summary['by_level'].get(level, 0) + 1
            
            job_role = data.get('job_role', 'Unknown')
            summary['by_job_role'][job_role] = summary['by_job_role'].get(job_role, 0) + 1
        
        return summary

    def analyze_graph_parameters(self, extracted_items: list, user_query: str) -> dict:
        """Use OpenAI to determine relevant_roles, info_level, and resource_type for graph generation."""
        prompt = f'''
Given the following extracted Adobe course/certificate names and the user query, select the most relevant job roles (from: Developer, Business Practitioner, Architect, All; you may select more than one), the appropriate info level for a graph visualization (choose one: low, medium, high), and the resource type (choose one: course, certificate, both).

Return a JSON object with:
- "relevant_roles": list of roles
- "info_level": one of "low", "medium", "high"
- "resource_type": one of "course", "certificate", "both"

Extracted items: {extracted_items}
User query: "{user_query}"

Return format:
{{
  "relevant_roles": ["Developer"],
  "info_level": "medium",
  "resource_type": "both"
}}
'''
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing Adobe learning pathways and graph visualization needs."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.1
            )
            content = response.choices[0].message.content.replace("`", "").replace("json", "")
            print(f"AI-selected relevant_roles/info_level/resource_type: {content}")
            if content and content.strip().startswith('{'):
                result = json.loads(content.strip())
                valid_roles = {"Developer", "Business Practitioner", "Architect", "All"}
                roles = [r for r in result.get("relevant_roles", []) if r in valid_roles]
                info_level = result.get("info_level", "medium")
                if info_level not in ["low", "medium", "high"]:
                    info_level = "medium"
                resource_type = result.get("resource_type", "both")
                if resource_type not in ["course", "certificate", "both"]:
                    resource_type = "both"
                return {"relevant_roles": roles or ["All"], "info_level": info_level, "resource_type": resource_type}
            else:
                return {"relevant_roles": ["All"], "info_level": "medium", "resource_type": "both"}
        except Exception as e:
            print(f"Error in analyze_graph_parameters: {e}")
            return {"relevant_roles": ["All"], "info_level": "medium", "resource_type": "both"}

def main():
    """Main function for the command-line interface."""
    print("Adobe Resources Expert - Agentic RAG (Command Line)")
    print("=" * 60)
    
    # Initialize RAG system
    print("Initializing Adobe Resources Expert...")
    rag_system = SimpleAgenticRAG()
    
    # Show resource summary
    summary = rag_system.get_resource_summary()
    print(f"\nResource Summary:")
    print(f"   Total Resources: {summary['total_resources']}")
    print(f"   Types: {', '.join(summary['by_type'].keys())}")
    print(f"   Levels: {', '.join(summary['by_level'].keys())}")
    
    print("\nExample questions:")
    print("   - What Adobe Analytics courses are available?")
    print("   - Tell me about Adobe Commerce certifications")
    print("   - What are the prerequisites for Expert level certifications?")
    print("   - How much do Adobe certifications cost?")
    print("   - Which courses are best for beginners?")
    print("   - Can you compare the different levels of Adobe certifications?")
    
    print("\n" + "=" * 60)
    
    # Chat loop
    while True:
        try:
            query = input("You: ").strip()
            
            if query.lower() in ['quit', 'exit', 'bye', 'q']:
                print("Goodbye! Happy learning!")
                break
            
            if query.lower() in ['history', 'h']:
                print("\nChat History:")
                for i, msg in enumerate(rag_system.chat_history[-5:], 1):
                    role = "User" if msg.role == "user" else "Assistant"
                    print(f"{i}. {role}: {msg.content[:100]}...")
                continue
            
            if query.lower() in ['summary', 's']:
                summary = rag_system.get_chat_summary()
                if summary:
                    print(f"\nConversation Summary: {summary}")
                else:
                    print("\nNo conversation history to summarize.")
                continue
            
            if query.lower() in ['clear', 'c']:
                rag_system.chat_history = []
                print("\nChat history cleared!")
                continue
            
            if not query:
                continue
            
            # Add user message to history
            rag_system.add_message("user", query)
            
            print("Thinking with advanced reasoning...")
            response = rag_system.get_response(query)
            
            # Add assistant response to history
            rag_system.add_message("assistant", response)
            
            print(f"\nAssistant: {response}")
            
        except KeyboardInterrupt:
            print("\nGoodbye! Happy learning!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main() 