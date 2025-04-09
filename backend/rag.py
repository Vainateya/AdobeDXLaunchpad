from openai import OpenAI
from documents import *
from graph_utils import *
from utils import *
import ast
import inspect
from enum import Enum
import re
import ast

class Bucket(Enum):
    IRRELEVANT = 1
    GENERAL = 2
    GRAPH = 3


COURSE_LEVELS = ['Foundations', 'Professional', 'Expert']
CERT_LEVELS = ['Professional', 'Expert', 'Master']


def extract_resources(text):
    """
    Parameters:
        text (str): The body of text containing the substring.
        
    Returns:
        list: The extracted list of strings, or None if not found.
    """
    pattern = r"<p hidden><strong>Resources Listed</strong>:\s*(\[[^\]]*\])</p>"

    match = re.search(pattern, text)
    if match:
        list_str = match.group(1)
        # Optionally, convert the string representation of the list to an actual list:
        resources = ast.literal_eval(list_str)
        return resources
    else:
        return []

def extract_resources_relations(text) -> tuple[str, str]:
    """
    Parameters:
        text (str): The body of text containing the substring.
        
    Returns:
        list: The extracted list of strings, or None if not found.
    """
    pattern = r"<p hidden><strong>Resources Ordered</strong>:\s*(\[[^\]]*\])</p>"

    match = re.search(pattern, text)
    if match:
        list_str = match.group(1)
        # Optionally, convert the string representation of the list to an actual list:
        resource_items = ast.literal_eval(list_str)
        resources = []
        for resource_items in resource_items:
            resources.append(resource_items.split("=>"))
        return resources
    else:
        return []

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
        self.role = Bucket.IRRELEVANT

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
        G = get_specific_graph(courses, certificates, relevant_roles = graph_args[0], info_level = graph_args[1], starting_nodes = graph_args[2])
        print("G", len(G), graph_args)
        if len(G) == 0:
            graph_args[0] = ["All"]
            G = get_specific_graph(courses, certificates, relevant_roles = graph_args[0], info_level = graph_args[1], starting_nodes = graph_args[2])
            print("REDO: G", len(G), graph_args)
        if len(G) == 0:
            graph = nx.DiGraph()
            return graph, None, None

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

    def generate_response(self, query, documents: str):
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
            <p>Ensure the plan follows logical <strong>prerequisite dependencies</strong>, starting with foundational courses and leading to expert certifications. <strong>ONLY</strong> refer to courses from the Retrieved Course Information section</p> 

            <h2> Output Format: HTML</h2>
            <ul>
                <li>Print which Role you are acting under.</li>
                <li>Wrap paragraphs in <code>&lt;p&gt;</code></li>
                <li>Use <code>&lt;h2&gt;</code> and <code>&lt;h3&gt;</code> for sections</li>
                <li>Use <code>&lt;ul&gt;&lt;li&gt;</code> for lists</li>
                <li>End any responses with a list of the course and certification names mentioned in your response up until now, formatted as "<p hidden><strong>Resources Listed</strong>:['Resource 1', 'Resource 2', ...]</p>" If no resources are listed, output <p><strong>Resources Listed</strong>:[]</p>" </li>
                <li>If any resource in "Resources Listed" should be followed in a specific order, please indicate that relationship in pairs formatted as "<p hidden><strong>Resources Ordered</strong>:['Resource 1 => Resource 2', 'Resource 2 => Resource 3', ...]</p>" If no resources are listed or there are no such relations, output <p><strong>Resources Ordered</strong>:[]</p>"</li>
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
            error_str = f"An error occurred while generating a response: {e}"
            print(error_str)
            return error_str
        
    def generate_general_response(self, query, documents: str):
        chat_history_text = self.format_chat_history()
        prompt = f"""
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
            error_str = f"An error occurred while generating a response: {e}"
            print(error_str)
            return error_str

    def generate_graph_args(self, query:str):
        chat_history_text = self.format_chat_history(html=False)

        prompt = f"""
            YOU ARE AN ADOBE COURSE/CERTIFICATE RECCOMENDATION BOT. 
            HOWEVER, YOU WILL ONLY USE THE SUPPORTING COURSE INFORMATION AND DOCUMENTS WHEN A USER ASKS FOR A COURSE RECCOMENDATION OR WHEN THEIR QUERY IS SPECIFIC. 
            IN MOST CASES YOU WILL ASK FOLLOWUP QUESIONS TO GAUGE THEIR INTERESTS BETTER. 
                            
            Previous Conversation: "{chat_history_text}"
            User Query: "{query}"
            Instructions: Using the Previous Conversation and User Query, output three metrics - job roles and information level. 
            For job roles, there are four options. Business Practitioners are responsible for designing, executing, and managing marketing campaigns using Adobe Experience Cloud solutions. They should have a foundational understanding of Adobe’s digital marketing solutions, as well as experience in marketing and advertising. The Business Practitioner certification validates their ability to effectively use Adobe’s digital marketing solutions to achieve business objectives. Developers are responsible for implementing and integrating Adobe Experience Cloud solutions into an organization’s technology stack. They should have experience in software development and proficiency in web technologies, such as HTML, CSS, JavaScript, and RESTful APIs. The Developer certification validates their ability to effectively implement and customize Adobe’s digital marketing solutions to meet business requirements. Architects are responsible for designing and implementing enterprise-grade solutions using Adobe Experience Cloud solutions. All indicates they may be interested in each of the previous three roles. Select the roles the user is interested in across "Developer, Business Practitioner, Architect, or 'All'.
 
            For information level, select how descriptive you think the user wants the results to be from "low", "medium", "high".

            Output your answer in the form [JOB_ROLES, INFORMATION_LEVEL], where JOB_ROLES is a list of strings, INFORMATION_LEVEL is a string.

            Example Output: [["All"], "medium"]
            Example Output: [["Developer", "Architect"], "high"]

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
            print("parsed_response", parsed_response)
            return parsed_response
        except Exception as e:
            return f"An error occurred while generating a response: {e}"
    
    def run_rag_pipeline(self, query: str, courses, certificates, top_k: int = 5) -> str:
        """Runs the full RAG pipeline: retrieves documents and generates a response."""
        bucket = self.find_bucket(query)
        if bucket == Bucket.IRRELEVANT:
            return "I don't understand what you said. Can you please ask something related to Adobe?", None
        elif bucket == Bucket.GENERAL:
            retrieved_docs = self.retrieve_documents(query, top_k)
            response = self.generate_general_response(query, retrieved_docs)
            return response, None
        else:
            retrieved_docs = self.retrieve_documents(query, top_k)
            response = self.generate_response(query, retrieved_docs)

            resource_names = extract_resources(response)
            resource_relations = extract_resources_relations(response)

            print("query", query)
            print("response", response)
            print("resource_names", resource_names)
            print("resource_relations", resource_relations)
            max_resources = 5

            if resource_names:
                job_roles, info_level = self.generate_graph_args(query)
                if info_level != "high":
                    max_resources = 1
                
                graph_args = [job_roles, info_level, resource_names[:max_resources]]
                graph, context, category = self.retrieve_graph(query, courses, certificates, graph_args)
            else:
                graph = nx.DiGraph()
                category = None
            return response, graph

    def run_graph_rag_pipeline(self, query, courses, certificates):
        category = self.get_category(query)
        response = self.generate_response_based_on_category(query, category, courses, certificates)
        return response

    def grouper(self, query):
        # Construct the prompt
        # Structured prompt with HTML output

        chat_history_text = self.format_chat_history()

        prompt = f"""
        YOU ARE AN ADOBE COURSE/CERTIFICATE RECOMMENDATION BOT. YOUR JOB IS TO IDENTIFY WHICH CATEGORY A USER'S CURRENT REQUEST FALLS UNDER.

        A USER QUERY IS CATEGORIZED INTO ONE OF THE FOLLOWING TYPES:

        1. **Irrelevant Request**  
        The query is not related to Adobe courses or certificates, or is general conversation with no actionable request.

        **Examples:**
        - "What's the weather like today?"
        - "Tell me a joke."
        - "Can you help me fix my printer?"

        2. **General Request**  
        The query is exploratory, asking for information about Adobe programs, courses, or certificates without requesting a specific learning path.

        **Examples:**
        - "What types of courses does Adobe offer?"
        - "Can you tell me more about the Adobe Analytics Professional course?"
        - "Are there any certificates for digital marketing?"

        3. **Modifying or Creating a Course Graph/Trajectory**  
        The user wants to receive a recommended course/certificate path or make changes to a previously suggested learning trajectory.

        **Examples:**
        - "What’s the best path to reach Adobe Analytics Expert?"
        - "I already completed Adobe Commerce Foundations, what should I take next?"
        - "Can you help me update my learning plan for a Master certificate in Adobe Analytics?"

        We will provide the previous conversation history and user query below.

        **Previous Conversation:**  
        "{chat_history_text}"

        **User Query:**  
        "{query}"

        Respond with ONLY a number: `1`, `2`, or `3` — indicating the category of the request.  
        No additional words, explanations, or symbols.
        """
        try:
            self.add_to_history("user", query)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                seed=42
            )
            assistant_response = response.choices[0].message.content
            self.add_to_history("assistant", assistant_response)
            return assistant_response
        except Exception as e:
            error_str = f"An error occurred while generating a response: {e}"
            print(error_str)
            return error_str
    
    def find_bucket(self, query: str):
        response = self.first_pass(query).strip()

        if "1" in response:
            self.role = Bucket.IRRELEVANT
        elif "2" in response:
            self.role = Bucket.GENERAL
        elif "3" in response:
            self.role = Bucket.GRAPH
        else:
            print("Warning: Unable to classify query response properly.")
            self.role = Bucket.IRRELEVANT  # fallback
        return self.role    
