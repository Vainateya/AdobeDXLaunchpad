from openai import OpenAI
from documents import *
from graph_utils import *
from utils import *
import ast
import inspect

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

    def retrieve_graph(self, query, courses, certificates, graph_args, top_k = 1):
        category = self.document_store.get_category_from_best_document(query)
        # starting_node = {
        #     'category': 'Adobe Commerce',
        #     'level': 'Foundations',
        #     'type': Course
        # }
        # G = get_specific_graph(courses, certificates, relevant_roles = ['All', 'Business Practitioner'], info_level = 'medium', starting_node = starting_node)
        G = get_specific_graph(courses, certificates, relevant_roles = graph_args[0], info_level = graph_args[1], starting_node = graph_args[2])
        nodes, edges = graph_to_2d_array(G)
        context = get_full_string(nodes, edges)
        return G, context, category

    def format_chat_history(self, html=True) -> str:
        """Formats the chat history for including in the prompt."""
        def format_text(role, s, html=True):
            if html:
                return f"<p><strong>{role}</strong>{s}</p>\n"
            else:
                return f"{role}:{s}\n"

        chat_history_text = ""
        for entry in self.chat_history:
            if entry["role"] == "user":
                chat_history_text += format_text(entry["role"], entry['content'], html=html)
            elif entry["role"] == "assistant":
                chat_history_text += format_text(entry["role"], entry['content'], html=html)
        return chat_history_text
    
    def format_chat_history_nohtml(self) -> str:
        """Formats the chat history for including in the prompt."""
        chat_history_text = ""
        for entry in self.chat_history:
            if entry["role"] == "user":
                chat_history_text += f"User: {entry['content']}\n"
            elif entry["role"] == "assistant":
                chat_history_text += f"Assistant: {entry['content']}\n"
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
        

    def generate_graph_args(self, query:str):
        chat_history_text = self.format_chat_history(html=False)

        prompt = f"""
            YOU ARE AN ADOBE COURSE/CERTIFICATE RECCOMENDATION BOT. 
            HOWEVER, YOU WILL ONLY USE THE SUPPORTING COURSE INFORMATION AND DOCUMENTS WHEN A USER ASKS FOR A COURSE RECCOMENDATION OR WHEN THEIR QUERY IS SPECIFIC. 
            IN MOST CASES YOU WILL ASK FOLLOWUP QUESIONS TO GAUGE THEIR INTERESTS BETTER. 
                            
            Previous Conversation: "{chat_history_text}"
            User Query: "{query}"
            Instructions: Using the Previous Conversation and User Query, output three metrics - job roles and information level. 
            For job roles, select the roles the user is interested in across "Developer, Business Practitioner, Architect, or 'All'. 
            For information level, select how descriptive you think the user wants the results to be from "low", "medium", "high".
            For category, select which category relates best to the user's query from the following options: "All", Adobe Analytics", "Adobe Commerce", "Adobe Customer Journey Analytics", "Adobe Experience Cloud", "Adobe Experience Manager", "Adobe Experience Platform", "Adobe Firefly", "Adobe GenStudio", "Adobe Journey Optimizer", "Adobe Marketo Engage", "Adobe Mix Modeler", "Adobe Real-Time CDP", "Adobe Target", "Adobe Workfront"
            For skill level, select which skill level corresponds to the user's current skill level the user would like begin with, 0 for beginning with the fundamentals, 1 for skipping to intermediete and 2 for skipping to expert: 0,1,2
            For resource, select which type of resource the user would be the most interested in, either a course to learn the fundamentals, or a certification, amongst the following options: "Course", "Certification"

            Output your answer in the form [JOB_ROLES, INFORMATION_LEVEL, CATEGORY, SKILL_LEVEL, RESOURCE], where JOB_ROLES is a list of strings, INFORMATION_LEVEL is a string, CATEGORY is a string, SKILL_LEVEL is an integer, and RESOURCE is a string.

            Example Output: [["All"], "medium", "Adobe Commerce", 1, "Certification"]
            Example Output: [["Developer", "Architect"], "high", "Adobe Analytics", 0, "Course"]

            Output:
            """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7
            )
            assistant_response = response.choices[0].message.content
            parsed_response = ast.literal_eval(assistant_response)

            job_roles = parsed_response[0]
            info_level = parsed_response[1]
            category = parsed_response[2]
            level_num = parsed_response[3]
            type_str = parsed_response[4]

            if type_str == "Course":
                level = ['Foundations', 'Professional', 'Expert'][level_num]
            else:
                level = ['Professional', 'Expert', 'Master'][level_num]

            start_node = {
                "category": category,
                "level": level,
                "type": Course if type_str == "Course" else Certificate,
            }

            graph_args = [job_roles, info_level, start_node]

            print("prompt", prompt)
            print("assistant_response", graph_args)

            return graph_args
        except Exception as e:
            return f"An error occurred while generating a response: {e}"
    
    def run_rag_pipeline(self, query: str, courses, certificates, top_k: int = 5) -> str:
        """Runs the full RAG pipeline: retrieves documents and generates a response."""
        retrieved_docs = self.retrieve_documents(query, top_k)
        graph_args = self.generate_graph_args(query)
        graph, context, category = self.retrieve_graph(query, courses, certificates, graph_args)
        response = self.generate_response(query, retrieved_docs, context)
        return response, category, graph

    def run_graph_rag_pipeline(self, query, courses, certificates):
        category = self.get_category(query)
        response = self.generate_response_based_on_category(query, category, courses, certificates)
        return response