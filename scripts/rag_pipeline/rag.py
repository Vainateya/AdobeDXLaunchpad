from openai import OpenAI
from documents import *
from AdobeDXLaunchpad.dependency_graph.graph_utils import *
from utils import *

class BasicRAG:
    """
    A default RAG (Retrieval-Augmented Generation) pipeline that retrieves relevant documents 
    from ChromaDB based on a user query and generates a response using OpenAI's API.
    """
    
    def __init__(self, document_store: DocumentStore, model: str = "gpt-4o-mini"):
        self.document_store = document_store
        self.model = model
        self.client = OpenAI()
        self.chat_history = [] 
    
    def add_to_history(self, role: str, content: str):
        """Appends a new entry to the chat history."""
        self.chat_history.append({"role": role, "content": content})
    
    def retrieve_documents(self, query: str, top_k: int = 5):
        """Fetches the top-k most relevant documents from the document store and formats metadata."""
        retrieved_docs = self.document_store.query_documents(query_text=query, top_k=top_k)
        # Format all metadata dynamically
        context = "\n\n".join(
            [
                f"<h3>{doc['metadata']['title']}</h3>" +
                "".join(f"<p><strong>{key.replace('_', ' ').title()}:</strong> {value}</p>"
                        for key, value in doc['metadata'].items() if key != "title")
                for doc in retrieved_docs
            ]
        )
        return context

    def retrieve_graph(self, query, courses, certificates, top_k = 1):
        category = self.document_store.get_category_from_best_document(query)
        G = create_graph(category, courses, certificates)
        nodes, edges = graph_to_2d_array(G)
        context = get_full_string(nodes, edges)
        return context
    
    def format_chat_history(self) -> str:
        """Formats the chat history for including in the prompt."""
        chat_history_text = ""
        for entry in self.chat_history:
            if entry["role"] == "user":
                chat_history_text += f"<p><strong>User:</strong> {entry['content']}</p>\n"
            elif entry["role"] == "assistant":
                chat_history_text += f"<p><strong>Assistant:</strong> {entry['content']}</p>\n"
        return chat_history_text

    def generate_response(self, query, documents: str, graph: str):
        # Construct the prompt
        # Structured prompt with HTML output

        chat_history_text = self.format_chat_history()

        prompt = f"""

            YOU ARE AN ADOBE COURSE/CERTIFICATE RECCOMENDATION BOT. 
            HOWEVER, YOU WILL ONLY USE THE SUPPORTING COURSE INFORMATION AND DOCUMENTS WHEN A USER ASKS FOR A COURSE RECCOMENDATION OR WHEN THEIR QUERY IS SPECIFIC. 
            IN MOST CASES YOU WILL ASK FOLLOWUP QUESIONS TO GAUGE THEIR INTERESTS BETTER. 
            
            <p><strong>Role Assignment:</strong></p>
            <ul>
                <li>If the query is <strong>too vague or seeks general career advice</strong>, act as a <strong>Career Counselor</strong>. Ignore the documents and ask a <strong>clarifying question</strong> to guide the user toward a concrete query.</li>
                <li>If the query <strong>asks about courses</strong> or Adobe offerings, act as a <strong>Course Explainer</strong>. Summarize the relevant courses from the retrieved <strong>documents</strong> and recommend the <strong>best fit course</strong> for the user's needs.</li>
                <li>If the user <strong>wants a structured learning plan</strong>, act as a <strong>Career Planner</strong>. Use both <strong>documents</strong> and the <strong>graph</strong> to outline a <strong>logical learning trajectory</strong> with prerequisites.</li>
            </ul>

            <h2>Previous Conversation:</h2>
            {chat_history_text}
            
            <h2>User Query:</h2>
            <p>{query}</p>
            
            <h2>Retrieved Course Information:</h2>
            <p>{documents}</p>
            
            <h2>Course Dependency Graph:</h2>
            <p>{graph}</p>
            
            <p><strong>Instructions:</strong> Based on the query, follow the appropriate role:</p>
            
            <h3>1 Career Counselor Role</h3>
            <p>If the query is vague or asks for career advice, <strong>ignore the retrieved documents</strong> and ask a clarifying question to help the user refine their request.</p>
            <p>For example, ask: <em>"Would you like a course recommendation or a full learning trajectory?"</em></p>

            <h3>2 Course Explainer Role</h3>
            <p>If the user asks about courses, <strong>summarize the most relevant courses</strong> from the retrieved documents.</p>
            <p>At the end, recommend the <strong>best course</strong> based on:</p>
            <ul>
                <li>The user's <strong>career goals</strong></li>
                <li>Their <strong>educational background</strong></li>
                <li>Their <strong>current skills</strong></li>
                <li>Course metadata (difficulty level, prerequisites, topic)</li>
            </ul>

            <h3>3 Career Planner Role</h3>
            <p>If the user asks for a <strong>course trajectory</strong>, use both the <strong>documents</strong> and the <strong>graph</strong> to create a step-by-step learning plan.</p>
            <p>Ensure the plan follows logical <strong>prerequisite dependencies</strong>, starting with foundational courses and leading to expert certifications.</p>

            <h2> Output Format: HTML</h2>
            <ul>
                <li>Wrap paragraphs in <code>&lt;p&gt;</code></li>
                <li>Use <code>&lt;h2&gt;</code> and <code>&lt;h3&gt;</code> for sections</li>
                <li>Use <code>&lt;ul&gt;&lt;li&gt;</code> for lists</li>
            </ul>
            
            <h2> Generate Your Response Below:</h2>
            """

        try:
            # Add the user query to the chat history
            self.add_to_history("user", query)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7
            )
            assistant_response = response.choices[0].message.content
            self.add_to_history("assistant", assistant_response)
            return assistant_response
        except Exception as e:
            return f"An error occurred while generating a response: {e}"
    
    def run_rag_pipeline(self, query: str, courses, certificates, top_k: int = 5) -> str:
        """Runs the full RAG pipeline: retrieves documents and generates a response."""
        retrieved_docs = self.retrieve_documents(query, top_k)
        graph = self.retrieve_graph(query, courses, certificates)
        response = self.generate_response(query, retrieved_docs, graph)
        return response

    def run_graph_rag_pipeline(self, query, courses, certificates):
        category = self.get_category(query)
        response = self.generate_response_based_on_category(query, category, courses, certificates)
        return response