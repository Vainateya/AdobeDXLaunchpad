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

def extract_dict_from_string(s: str):
    pattern = re.compile(r'\{.*?\}', re.DOTALL)
    matches = pattern.finditer(s)

    for match in matches:
        dict_str = match.group()
        try:
            extracted_dict = ast.literal_eval(dict_str)
            if isinstance(extracted_dict, dict):
                return extracted_dict
        except (ValueError, SyntaxError):
            continue  # Try next match if current one fails parsing

    raise ValueError("No valid dictionary found in the provided string.")

class BasicRAG:
    """
    A default RAG (Retrieval-Augmented Generation) pipeline that retrieves relevant documents 
    from ChromaDB based on a user query and generates a response using OpenAI's API.
    """
    
    def __init__(self, document_store: DocumentStore, supplement_store: DocumentStore, model: str = "gpt-4o-mini"):
        self.document_store = document_store
        self.supplement_store = supplement_store
        self.model = model
        self.client = OpenAI()
        self.chat_history = [] 
        self.role = Bucket.IRRELEVANT
        self.past_graphs: tuple[str, list[str]] = [] # {user_query, [coursename, coursename, ...]}
        self.current_graph = [] # {}
        self.current_nx_graph = nx.DiGraph()
        self.title2doc = {d['metadata']['title']: d for d in self.document_store.get_all_documents()}
        self.temperature = 0

    def format_docs_context(self, docs):
        if docs:
            for doc in docs:
                print(doc['metadata'].keys())
        
        return "\n\n".join(
            [
                f"<h3>{doc['metadata']['title']}</h3> <p><strong>THIS IS A {doc['metadata']['type'].upper()}</strong></p>" +
                "".join(
                    f"<p><strong>{key.replace('_', ' ').title()}:</strong> {value}</p>"
                    for key, value in doc['metadata'].items() if key != "title"
                )
                for doc in docs
            ]
        )

    def add_to_history(self, role: str, content: str):
        """Appends a new entry to the chat history."""
        self.chat_history.append({"role": role, "content": content})

    def retrieve_documents(self, query: str, top_k: int = 5, exclude_supplement: bool = False, omit_titles = []):
        """Fetches the top-k most relevant documents from the document store and formats metadata.
        If exclude_supplement=True, excludes supplemental docs (only uses course/cert content)."""

        all_docs = self.document_store.query_documents(query_text=query, top_k=top_k)
        filtered_docs = [doc for doc in all_docs if doc['metadata']['title'] not in omit_titles]

        if not exclude_supplement:
            supplement_docs = self.supplement_store.query_documents(query_text=query, top_k=top_k)
            filtered_docs += supplement_docs  # same as all_docs = all_docs + supplement_docs

        print("Inner context")
        context = self.format_docs_context(filtered_docs)

        return context, filtered_docs

    def retrieve_graph(self, courses, certificates, sources):
        G = get_llm_graph(courses, certificates, sources)
        if len(G) == 0:
            graph = nx.DiGraph()
            return graph
        return G

    def format_chat_history(self, html=True) -> str:
        """Formats the chat history for including in the prompt."""
        def format_text(role, s, html=True):
            if html:
                return f"<p><strong>{role} previously said: </strong>{s}</p>\n"
            else:
                return f"{role} previously said::{s}\n"

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
                chat_history_text += f"User previously said: {entry['content']}\n"
            elif entry["role"] == "assistant":
                chat_history_text += f"Assistant previously said: {entry['content']}\n"
        return chat_history_text

    def format_past_graphs(self, graphs: list[str]):
        out = ''
        for i, (query, graph) in enumerate(graphs):
            out +=  f"At time {i}: User asked '{query}'. Resources in the resource graph at that time were {str(graph)}\n'"
        return out

    def generate_new_graph(self, query, cur_resources={}, history=[], resource_info=""):
        # Construct the prompt
        # Structured prompt with HTML output

        prompt = f'''You are a strict, rules-based trajectory planner that simulates valid learning pathways. Your task is to return a JSON object with one of the following operations: ADD, SUBTRACT, OVERHAUL, GO_BACK, or NO_CHANGE. You must only work with the resources explicitly listed in the 'Resources you should know about' section. Follow all rules below exactly. Violating any rule will make the response invalid.
        
        Only include the operation(s) you are instructed to perform. You may not mix operations unless explicitly allowed. For example, you may not combine ADD and OVERHAUL. GO_BACK must be an integer index into resource history.

        -----------------------------------
        RULES (MUST BE FOLLOWED EXACTLY)
        -----------------------------------

        - R1 (Course Progression): Courses follow this order — Foundations → Professional → Expert — within the SAME category.
        - R2 (Certificate Progression): Certificates follow this order — Professional → Expert → Master — within the SAME category.
        - R3 (Course → Certificate Cross-Dependencies): Courses may serve as prerequisites for certificates ONLY if the level and category match:
            - Foundations course → Professional certificate
            - Professional course → Expert certificate
            - Expert course → Master certificate
        - R4 (Same Category Required): Resources must share the same category to be part of a valid prerequisite chain. A Foundations course in Adobe Analytics does NOT unlock a Professional course in Adobe Commerce.
        - R5 (Job Role Filtering for Certificates): Certificates must match the user’s job role, or be tagged with the job role "All".
        - R6 (No Level Skipping): You must not skip levels. For example, you cannot go from a Foundations course directly to an Expert course.

        -----------------------------------
        RESOURCE DEFINITIONS
        -----------------------------------

        {resource_info}
        
        -----------------------------------
        OPERATIONS DEFINITION
        -----------------------------------

        - ADD: Only add new resources to the current graph from those listed in 'Resources you should know about'. Do not remove existing resources unless performing SUBTRACT simultaneously.
        - SUBTRACT: Only remove resources from the current graph. Do not add new resources unless paired with ADD.
        - OVERHAUL: Completely discard the current graph and replace it with a brand new valid trajectory. This operation cannot be combined with any other.
        - GO_BACK: Restore the resource graph to a previous state, specified by index from 'Resource History'. Do not include any resource names.
        - NO_CHANGE: Return the current graph unchanged. Use this when the user does not request modifications.
        
        -----------------------------------
        OUTPUT FORMAT
        -----------------------------------

        Your output must be a JSON object in the following possible formats:
        Option 1: Adding, Subtracting, or Both: 
        Response: {{
        "ADD": ["RESOURCE_1", "RESOURCE_2"],
        "SUBTRACT": ["RESOURCE_3"],
        }}
        
        Option 2: OVERHAUL, no other operations
        Response: {{
        "OVERHAUL": ["RESOURCE_4", "RESOURCE_5"],
        }}
        
        Option 3: GO_BACK, no other operations
        Response: {{
            "GO_BACK": TIME_STEP,
        }}

        Option 4: NO_CHANGE, no other operations
        Response: {{
        "NO_CHANGE": []
        }}

        -----------------------------------
        EXAMPLES
        -----------------------------------

        # Example 1: Valid Overhaul
        User Query: "Build me a full Adobe Analytics journey."
        Response:
        {{
        "OVERHAUL": ["Adobe Analytics Foundations", "Adobe Analytics Business Practitioner Professional", "Adobe Analytics Business Practitioner Expert"]
        }}

        # Example 2: Invalid category jump
        User Query: "Add Adobe Commerce Professional course after Adobe Analytics Foundations."
        Response:
        {{
        "ADD": []  # Invalid. Cannot add Professional course from a different category.
        }}

        Now respond to the following:
        User Query: {query}
        Current Resources in Graph: {str(cur_resources)}
        Resource History: {self.format_past_graphs(history)}
        '''


        try:
            print("Graph Generation Prompt:", prompt)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature
            )
            assistant_response = response.choices[0].message.content
            return assistant_response
        except Exception as e:
            error_str = f"An error occurred while generating a response: {e}"
            print(error_str)
            return error_str
        
    def generate_general_response(self, query, documents: str, user_profile, bucket, graph_str_raw=""):
        chat_history_text = self.format_chat_history()
        # print("QUERY", query)
        # print("documents", documents)
        # print("user_profile", user_profile)
        # print("graph_str_raw", graph_str_raw)

        graph_str = ""
        if graph_str_raw:
            graph_str = f'''
            <h3>Resource Graph: This is the user's current learning pathway, including courses, study guides, and certifications. Use this if the user's query relates to modify the graph or questions about the courses on the graph</h3>
            <p>{graph_str_raw}</p>
            '''

        prompt = f"""
        <h1>Adobe Learning Assistant</h1>

        <h2><strong>Your Role</strong></h2>
        <p>You are an intelligent assistant for Adobe users seeking learning guidance. Based on the user’s query, resource graph, user profile, and document context, respond in a way that is helpful, well-reasoned, and matches the user’s level and goals.</p>

        <h2><strong>How to Use Context</strong></h2>
        <ol>
        <li><strong>Use the Resource Graph if it is present</strong> — This is the user’s current learning plan. Prefer it over external course documents for accuracy and continuity.</li>
        <li>Use the <strong>Retrieved Documents</strong> only when no relevant learning pathway is defined in the graph.</li>
        <li>Use the <strong>Chat History</strong> to understand the user’s evolving goals and prior interests.</li>
        <li>Use the <strong>User Profile</strong> (if available) to tailor tone, role alignment, and course level recommendations.</li>
        </ol>

        <h2><strong>Context</strong></h2>

        <h3>User Profile:</h3>
        <p>{user_profile}</p>

        <h3>Chat History:</h3>
        <p>{chat_history_text}</p>

        <h3>Current User Query:</h3>
        <p>{query}</p>

        {graph_str}

        <h3>Retrieved Course & Certificate Documents:</h3>
        <p>{self.format_docs_context(documents)}</p>

        <h2><strong>Instructions</strong></h2>

        <p>Use the following decision logic to respond:</p>
        <ul>
        <li><strong>If the query is vague</strong>: Ask a clear, helpful clarifying question to guide the user toward a learning goal.</li>
        <li><strong>If the query asks for a course or certification recommendation</strong>: Use the Resource Graph if available. Otherwise use the retrieved documents. Always explain your reasoning, referencing level, role, or objectives.</li>
        <li><strong>If the query asks about specific programs</strong>: Summarize those programs from the most reliable source (Graph → Docs), and explain fit based on user role, level, or prior steps.</li>
        <li><strong>If the query is programmatic or logistical</strong>: Answer directly from document context.</li>
        </ul>

        <h2><strong>Rules You Must Follow</strong></h2>
        <ul>
        <li><strong>Only recommend resources listed in the graph if it is present.</strong></li>
        <li><strong>Never hallucinate course or certificate names.</strong> Only use names exactly as provided.</li>
        <li><strong>Clearly distinguish between courses and certificates.</strong></li>
        </ul>

        <h2><strong>Response Format</strong></h2>
        <ul>
        <li>Use <code>&lt;p&gt;</code> for explanations.</li>
        <li>Use <code>&lt;h2&gt;</code> and <code>&lt;h3&gt;</code> for structure.</li>
        <li>Use bullet lists for course breakdowns.</li>
        </ul>

        <h2><strong>Now generate your response below:</strong></h2>

        """
        try:
            print("General Reponse Prompt:", prompt)
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                stream=True,
                temperature=self.temperature
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    yield token

            # assistant_response = response.choices[0].message.content
            # lines = assistant_response.splitlines()
            # assistant_response = "\n".join(lines[1:-1])
            # return assistant_response
        except Exception as e:
            error_str = f"An error occurred while generating a response: {e}"
            print(error_str)
            return error_str

    def update_graph_state(self, query, courses, certificates, user_profile):
        bucket = self.find_bucket(query)
        if bucket == Bucket.IRRELEVANT:
            return self.current_nx_graph
    
        rag_query = 'Current resources in graph: ' + ', '.join("'" + name + "'" for name in self.current_graph) + " User Query: " + query

        retrieved_docs = self.retrieve_documents(rag_query, top_k=10, exclude_supplement=True)[1]
        retrieved_docs.extend(
            self.retrieve_documents(query, top_k=5, exclude_supplement=True, omit_titles=[d['metadata']['title'] for d in retrieved_docs])[1]
        )
        current_docs = [self.title2doc[g] for g in self.current_graph if g in self.current_graph and 'Study Materials' not in g] + retrieved_docs
        resource_info = self.format_docs_context(current_docs)

        raw_graph_ops = self.generate_new_graph(query, self.current_graph, self.past_graphs, resource_info)
        graph_ops = extract_dict_from_string(raw_graph_ops)

        if 'ADD' in graph_ops:
            self.current_graph.extend([g for g in graph_ops['ADD'] if g in self.title2doc])
        if 'SUBTRACT' in graph_ops:
            self.current_graph = [source for source in self.current_graph if source not in graph_ops['SUBTRACT']]
        if 'OVERHAUL' in graph_ops:
            self.current_graph = [g for g in graph_ops['OVERHAUL'] if g in self.title2doc]
        if 'GO_BACK' in graph_ops:
            self.current_graph = self.past_graphs[int(graph_ops['GO_BACK'])][1]

        self.current_nx_graph = self.retrieve_graph(courses, certificates, self.current_graph)
        self.past_graphs.append((query, self.current_graph))
        return self.current_nx_graph

    def run_rag_pipeline_stream(self, query: str, courses, certificates, user_profile, top_k: int = 5):
        """
        Runs the RAG pipeline and yields assistant response tokens as a stream.
        This version is designed for use in Flask streaming endpoints.
        """
        bucket = self.find_bucket(query)
        print("BUCKET:", bucket)

        if bucket == Bucket.IRRELEVANT:
            yield "I’m not sure how to help with that one — but if you’re looking to grow your skills with Adobe, I’d be happy to guide you! Want to explore a specific product or learning path?"
            return

        # Add user message to chat history
        self.add_to_history("user", query)

        # Use existing graph if applicable
        if len(self.current_graph) > 0:
            _, e = graph_to_2d_array(self.current_nx_graph)
            graph_str = display_edges(e)
        else:
            graph_str = ""

        # Retrieve relevant docs
        retrieved_docs = self.retrieve_documents(query, top_k)[1]

        # Start the generator for streamed assistant response
        stream = self.generate_general_response(query, retrieved_docs, user_profile, 'Graph' if bucket == Bucket.GRAPH else 'General', graph_str_raw=graph_str)

        full_response = ""
        for chunk in stream:
            full_response += chunk
            yield chunk  # Stream to client

        # Save full response at the end
        self.add_to_history("assistant", full_response)

    def grouper(self, query):
        # Construct the prompt
        # Structured prompt with HTML output

        chat_history_text = self.format_chat_history()

        """
        1. Irrelevant Request  
        The query is completely, entirely irrelevent, then we should not entertain it!

        Examples:
        - "What's the weather like today?"
        - "Tell me a joke."
        - "Can you help me fix my printer?"
        """

        prompt = f"""
        YOU ARE AN ADOBE COURSE/CERTIFICATE RECOMMENDATION BOT. YOUR JOB IS TO IDENTIFY WHICH CATEGORY A USER'S CURRENT REQUEST FALLS UNDER.

        A USER QUERY IS CATEGORIZED INTO ONE OF THE FOLLOWING TYPES:

        1. Irrelevant Request  
        The query is completely, entirely irrelevent, then we should not entertain it!

        Examples:
        - "What's the weather like today?"
        - "Tell me a joke."
        - "Can you help me fix my printer?"

        2. General Request  
        In this case the query is a general question, giving a greeting, asking for information about Adobe programs, asking for information about you, what you do, courses, or certificates without requesting a specific learning path. 
        It could also ask general logistical and programmatic questions about Adobe courses and certifications. This will mostly rely on the user query. 

        Examples:
        - "Hello"
        - "What do you do?"
        - "Wha is this about?"
        - "What types of courses does Adobe offer?"
        - "Can you tell me more about the Adobe Analytics Professional course?"
        - "Are there any certificates for digital marketing?"
        - "How do I sign up for a certification?"
        - "Are exams available in languages other than English?"

        If user wants anything AT ALL to do with the current learning path, please default to the third option below:

        3. Modifying or Creating a Course Graph/Trajectory
        The user wants to receive a recommended course/certificate path or make changes to a previously suggested learning trajectory. This will rely on the user query and previous conversation.

        Examples:
        - "What’s the best path to reach Adobe Analytics Expert?"
        - "I already completed Adobe Commerce Foundations, what should I take next?"
        - "Can you help me update my learning plan for a Master certificate in Adobe Analytics?"
        - "I am new to adobe commerce - what is a full learning journey, from start to finish?"
        
        ONLY USE THE CURRENT QUERY TO DETIRMINE THE CATEGORY, HOWEVER, CONSIDER THE QUERY IN CONTEXT OF HISTORY!

        Previous Conversation:  
        "{chat_history_text}"

        User Query:  
        "{query}"

        Respond with ONLY a number: `1`,`2`, or `3` — indicating the category of the request.  
        No additional words, explanations, or symbols.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                seed=42
            )
            assistant_response = response.choices[0].message.content
            return assistant_response
        except Exception as e:
            error_str = f"An error occurred while generating a response: {e}"
            print(error_str)
            return error_str
    
    def find_bucket(self, query: str):
        if self.chat_history and query in [self.chat_history[-1]['content'], self.chat_history[-2]['content']]:
            return self.role
        
        response = self.grouper(query).strip()

        # if "1" in response:
        #     self.role = Bucket.GENERAL
        # elif "2" in response:
        #     self.role = Bucket.GRAPH

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
