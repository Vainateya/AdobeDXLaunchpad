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

    def format_docs_context(self, docs):
        return "\n\n".join(
            [
                f"<h3>{doc['metadata']['title']}</h3> <p>THIS IS A {doc['metadata']['type'].upper()}</p>" +
                "".join(f"<p><strong>{key.replace('_', ' ').title()}:</strong> {value}</p>"
                        for key, value in doc['metadata'].items() if key != "title")
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

        if not exclude_supplement:
            supplement_docs = self.supplement_store.query_documents(query_text=query, top_k=top_k)
            all_docs += supplement_docs  # same as all_docs = all_docs + supplement_docs

        filtered_docs = [doc for doc in all_docs if doc['metadata']['title'] not in omit_titles]

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

    def format_past_graphs(self, graphs: list[str]):
        out = ''
        for i, (query, graph) in enumerate(graphs):
            out +=  f"At time {i}: User asked '{query}'. Resources in the resource graph at that time were {str(graph)}\n'"
        return out

    def generate_new_graph(self, query, cur_resources={}, history=[], resource_info=""):
        # Construct the prompt
        # Structured prompt with HTML output
        prompt = f"""
            You are a trajectory recommender whose job is, given a list of sources (course or certificate), output a valid trajectory that consists of a subset of those sources and that follow the below rules perfectly.

            Source Types - There are two types of sources:
            - Courses: format = <category> <level>
            - Certificates: format = <category> <job role> <level>

            Course Levels (in order):
            - Foundations → Professional → Expert

            A course can only require the previous level in the same category.
            - Example: Adobe Analytics Foundations → Adobe Analytics Professional (valid)
            - Adobe Analytics Foundations → Adobe Commerce Professional (invalid)

            Certificate Levels (in order):
            - Professional → Expert → Master

            Certificates have the same prerequisite structure as courses.
            - Job role is NOT used for prerequisites, but only show certificates matching the user's job role (or with job role All).

            Cross-Dependencies (Courses → Certificates) - Courses can be prerequisites for certificates if the category matches:
            - Foundations course → Professional certificate
            - Professional course → Expert certificate
            - Expert course → Master certificate

            Summary of Rules:
            - Match levels in order (no skipping) within the same category.
            - Prerequisites MUST be in the SAME category AND follow the level order.
            - Certificates must also match the user's job role (or be All).
            - Course → Certificate prerequisites allowed as per cross-dependency mapping.
            
            Resource History - The previous trajectories at previous iterations. Sometimes the user may want to reference an earlier trajectory, though sometimes they may not.

            Current Resouces in Graph - This is the current trajectory the user is discussing. The user may want to add resources, remove resources, or completely disregard the current trajectory and come up with a new one.
            You have the following operations:
            - ADD: ONLY add resources to the current graph from resources listed in 'Resources you should know about'. You can ONLY add resources matching the type(s) the user requests—courses, certificates, study guides, or a combination. You CANNOT remove any resources unless performing alongside a SUBTRACT operation. (e.g., "Can you add more certificates related to Python programming?")
            - SUBTRACT: ONLY remove resources from the current graph. This typically happens when the user explicitly requests fewer resources or asks to remove specific resources or resource types. You CANNOT add any resources unless performing alongside an ADD operation. (e.g., "Please remove any advanced-level courses." or "Take out the AWS certification.")
            - OVERHAUL: Disregard the current graph entirely and create a completely new graph - You MUST generate new nodes, this cannot return an empty list. Usually chosen when the user requests a fresh start or completely switches fields/topics. Cannot be combined with any other operation. (e.g., "Create a new graph focused exclusively on machine learning certifications.")
            - GO_BACK: Disregard the current graph and return to a previously saved graph state (timestep) stored in 'Resource History'. Unlike other operations, you must return an index (timestep), not a graph. Cannot be combined with other operations. (e.g., "Go back to the previous version of the graph," or "Undo the last change.")
            - NO_CHANGE: Used when the user does not request any changes. Return the current graph without alterations. Cannot be combined with other operations. (e.g., "What courses do I currently have?" or "Just show me my current graph.")

            Format the response to this as a set of the operations: {{ "OPERATION_1": ["COURSE_NAME_1", "COURSE_NAME_2", ...], "OPERATION_2": ["COURSE_NAME_3", "COURSE_NAME_4", ...] }}. If the operation is GO_BACK, return {{ GO_BACK: INDEX }}
            
            IT IS OF UTMOST IMPORTANCE THAT THESE RULES ARE FOLLOWED PERFECTLY WHEN OUTPUTTING A LIST OF NODES.
            
            <h2>Resources you should know about</h2>:
            <p>
            {resource_info}
            </p>

            Example:
            <h2>User Query:</h2>
            <p>This is too hard - can you give me easier classes?</p>
            <h2>Current Resouces in Graph</h2>
            <p>
            ["Adobe Analytics Foundations", "Adobe Analytics Business Practitioner Professional", "Adobe Analytics Business Practitioner Expert", "Adobe Analytics Architect Master"]
            </p>
            <h2>Resource History:</h2>
            <p>At time 0: User asked 'Give me a course trajectory to get good at Adobe Analytics'. Resources in the resource graph at that time were ['Adobe Analytics Foundations', 'Adobe Analytics Business Practitioner Professional', 'Adobe Analytics Business Practitioner Expert']</p>
            <h2>Response:</h2>
            <p>
            {{ "SUBTRACT": ["Adobe Analytics Foundations"] }}
            </p>
            
            The following is the true input:
            <h2>User Query:</h2>
            <p>{query}</p>
            <h2>Current Resouces in Graph</h2>
            <p>{str(cur_resources)}</p>
            <h2>Resource History:</h2>
            <p>{self.format_past_graphs(history)}</p>
            <h2>Response:</h2>
            """
        try:
            print("Prompt:", prompt)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7
            )
            assistant_response = response.choices[0].message.content
            return assistant_response
        except Exception as e:
            error_str = f"An error occurred while generating a response: {e}"
            print(error_str)
            return error_str
        
    def generate_general_response(self, query, documents: str, user_profile, graph_str_raw=""):
        chat_history_text = self.format_chat_history()
        print("QUERY", query)
        print("documents", documents)
        print("user_profile", user_profile)
        print("graph_str_raw", graph_str_raw)

        graph_str = ""
        if graph_str_raw:
            graph_str = f'''
            <h3>Resource Graph: This is the user's current learning pathway, including courses, study guides, and certifications. Use this if the user's query relates to modify the graph or questions about the courses on the graph</h3>
            <p>{graph_str}</p>
            '''

        prompt = f"""
        <!-- YOU ARE AN INTELLIGENT COURSE AND CERTIFICATE RECOMMENDATION ASSISTANT FOR ADOBE USERS. USE THE USER’S BACKGROUND, PREVIOUS PLAN HISTORY, AND CURRENT QUERY TO EITHER GENERATE RECOMMENDATIONS, EXPLAIN ADOBE COURSES, OR ASK CLARIFYING QUESTIONS. -->

        <h1>Adobe Learning Assistant</h1>

        <h2><strong>Here is all the context you need</strong></h2>

        <h3>This is the User Profile, please use it to make a tailored response:</h3>
        <p>{str(user_profile)}</p>

        <h3>This is the chat history, please use it to learn about where the conversation evolved from:</h3>
        {chat_history_text}

        <h3>Current User Query:</h3>
        <p>{query}</p>

        <h3>Retrieved Course & Certificate Documents, THIS IS IMPORTANT, this is all the relevant information from Adobe regarding this request. Please put your responses using this information:</h3>
        <p>{documents}</p>

        {graph_str}

        <h3>ENSURE THAT YOU ONY PULL INFORMATION FROM THE LEARNING PATHWAY AND DISREGARD DOCUMENTS IF LEARNING PATH IS PRESENT (ITS MORE TAILORED)</h3>

        <h2><strong>Instructions</strong></h2>

        <p>You are to act in the following ways depending on the user query and available context. Respond in HTML. Be thoughtful in your reasoning and helpful in your tone.</p>

        <p>If the query is programmatic in nature or has to do with general logistical or enrollment information/faqs, please answer to the best of the supporting documents above.</p>

        <p>If the query is vague or too general, ignore the documents and ask a clarifying question to understand what the user wants to learn or achieve.</p>
        <ul>
            <li>Example: “Would you like to explore a specific Adobe product or get a full learning path recommendation?”</li>
        </ul>

        <p>If the user asks about specific courses or programs, use <strong>documents + user profile</strong> to summarize and recommend one or more suitable courses.</p>
        <ul>
            <li>Consider: background, goals, current skills, job role, course level, and prerequisites.</li>
            <li>Explain why the recommendation fits.</li>
        </ul>

        <p>If the user is looking for a full learning trajectory or update to a plan, use the <strong>graph + documents + user profile</strong> to build a logical course sequence.</p>
        <ul>
            <li>Start from the user’s current level or completed courses.</li>
            <li>Ensure the sequence respects prerequisites and role alignment.</li>
        </ul>

        <h2><strong>Output Requirements (All in HTML):</strong></h2>
        <ul>
            <li>Clearly state which role you are acting under at the top.</li>
            <li>Use <code>&lt;p&gt;</code> for paragraphs.</li>
            <li>Use <code>&lt;h2&gt;</code>, <code>&lt;h3&gt;</code> for sectioning.</li>
            <li>Use <code>&lt;ul&gt;&lt;li&gt;</code> for lists.</li>
        </ul>

        <h2><strong>ENSURE THAT ANY COURSES YOU RECCOMEND ONLY COME FROM THE GRAPH IF ITS PROVIDED!</strong></h2>

        <h2><strong>ENSURE THAT YOU DONT MISLABEL COURSES AND CERTIFICATES!</strong></h2>

        <h2><strong>Generate Your Response Below:</strong></h2>
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
    
    def run_rag_pipeline(self, query: str, courses, certificates, user_profile, top_k: int = 5) -> str:
        """Runs the full RAG pipeline: retrieves documents and generates a response."""
        bucket = self.find_bucket(query)
        if bucket == Bucket.IRRELEVANT:
            default_str = "I don't understand what you said. Can you please ask something related to Adobe?"
            self.add_to_history("user", query)
            self.add_to_history("assistant", default_str)
            return default_str, nx.DiGraph()
        elif bucket == Bucket.GENERAL:
            print("BUCKET", bucket)
            if len(self.current_graph) > 0:
                _, e = graph_to_2d_array(self.current_nx_graph)
                graph_str = display_edges(e)
            else:
                graph_str = ""
        
            retrieved_docs = self.retrieve_documents(query, top_k)
            response = self.generate_general_response(query, retrieved_docs, user_profile, graph_str)
            self.add_to_history("user", query)
            self.add_to_history("assistant", response)
            return response, self.current_nx_graph
        else:
            #TODO: Integrate Chat History with graph history
            #TODO: Allow for manipulation of study guides

            rag_query = 'Current resources in graph: ' + ', '.join("'" + name + "'" for name in self.current_graph) + " User Query: " + query
            retrieved_docs = self.retrieve_documents(rag_query, top_k=10, exclude_supplement=True)[1]
            retrieved_docs.extend(self.retrieve_documents(query, top_k=5, exclude_supplement=True, omit_titles=[d['metadata']['title'] for d in retrieved_docs])[1])
            # print(query)
            # print(rag_query)
            # print("Retrieved docs:", [d['metadata']['title'] for d in retrieved_docs], "Current docs:", self.current_graph)
            # print("graph_list:", self.current_graph)

            # TODO: Handle Study materials
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

            if self.current_graph:
                graph = self.retrieve_graph(courses, certificates, self.current_graph)
                self.current_nx_graph = graph
                _, e = graph_to_2d_array(graph)
                graph_str = display_edges(e)
            else:
                graph = nx.DiGraph()
                graph_str = ""
            
            retrieved_docs = [doc for doc in retrieved_docs if doc['metadata']['title'] in self.current_graph]
            
            print("graph ops:", graph_ops)
            print("Current graph resources:", self.current_graph)
            self.past_graphs.append((query, self.current_graph))

            if len(self.current_graph) > 0:
                n, e = graph_to_2d_array(graph)
                graph_str = display_edges(e)
                self.current_nx_graph = graph

            response = self.generate_general_response(query, retrieved_docs, user_profile, graph_str)
            #response = "The graph is updated with your changes. Please let me know if you have questions about any of the courses or certificates!"
            self.add_to_history("user", query)
            self.add_to_history("assistant", response)
            return response, graph

    def grouper(self, query):
        # Construct the prompt
        # Structured prompt with HTML output

        chat_history_text = self.format_chat_history()

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
        In this case the query is a general question, asking for information about Adobe programs, courses, or certificates without requesting a specific learning path. 
        It could also ask general logistical and programmatic questions about Adobe courses and certifications. 

        Examples:
        - "What types of courses does Adobe offer?"
        - "Can you tell me more about the Adobe Analytics Professional course?"
        - "Are there any certificates for digital marketing?"
        - "How do I sign up for a certification?"
        - "Are exams available in languages other than English?"

        If user wants anything AT ALL to do with the current learning path, please default to the third option below:

        3. Modifying or Creating a Course Graph/Trajectory
        The user wants to receive a recommended course/certificate path or make changes to a previously suggested learning trajectory.

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

        Respond with ONLY a number: `1`, `2`, or `3` — indicating the category of the request.  
        No additional words, explanations, or symbols.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                seed=42
            )
            assistant_response = response.choices[0].message.content
            return assistant_response
        except Exception as e:
            error_str = f"An error occurred while generating a response: {e}"
            print(error_str)
            return error_str
    
    def find_bucket(self, query: str):
        response = self.grouper(query).strip()

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
