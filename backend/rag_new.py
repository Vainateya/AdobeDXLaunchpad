from openai import OpenAI
from documents import *
from graph_utils import *
from utils import *
import ast
import inspect
import tiktoken
import time

COURSE_LEVELS = ['Foundations', 'Professional', 'Expert']
CERT_LEVELS = ['Professional', 'Expert', 'Master']

import re
import ast

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
    
    def __init__(self, document_store: DocumentStore, model: str = "gpt-4o-mini"):
        self.document_store = document_store
        self.model = model
        self.client = OpenAI()
        self.chat_logs = [] 
        self.chat_summary: str = ""
        self.summarized_to = 0 # idx + 1
        self.past_graphs: tuple[str, list[str]] = [] # {user_query, [coursename, coursename, ...]}
        self.current_graph = [] # {}
        self.title2doc = {d['metadata']['title']: d for d in self.document_store.get_all_documents()}

    def format_docs_context(self, docs):
        return "\n\n".join(
            [
                f"<h3>{doc['metadata']['title']}</h3> <p>THIS IS OF TYPE {doc['metadata']['type']}, NO OTHER TYPE!!!</p>" +
                "".join(f"<p><strong>{key.replace('_', ' ').title()}:</strong> {value}</p>"
                        for key, value in doc['metadata'].items() if key != "title")
                for doc in docs
            ]
        )

    def retrieve_documents(self, query: str, top_k: int = 8, omit_titles = []):
        """Fetches the top-k most relevant documents from the document store and formats metadata."""
        retrieved_docs = self.document_store.query_documents(query_text=query, top_k=top_k)
        # Format all metadata dynamically
        filtered_docs = [doc for doc in retrieved_docs if doc['metadata']['title'] not in omit_titles]
        context = self.format_docs_context(filtered_docs)
        return context, filtered_docs
    
    def retrieve_graph(self, courses, certificates, sources):
        G = get_llm_graph(courses, certificates, sources)
        if len(G) == 0:
            graph = nx.DiGraph()
            return graph
        return G

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
            - OVERHAUL: Disregard the current graph entirely and create a completely new graph. Usually chosen when the user requests a fresh start or completely switches fields/topics. Cannot be combined with any other operation. (e.g., "Create a new graph focused exclusively on machine learning certifications.")
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
            start_time = time.time()
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

    def format_past_graphs(self, graphs: list[str]):
        out = ''
        for i, (query, graph) in enumerate(graphs):
            out +=  f"At time {i}: User asked '{query}'. Resources in the resource graph at that time were {str(graph)}\n'"
        return out

    def run_rag_pipeline(self, query: str, courses, certificates, top_k: int = 5) -> str:
        """Runs the full RAG pipeline: retrieves documents and generates a response."""

        rag_query = 'Current resources in graph: ' + ', '.join("'" + name + "'" for name in self.current_graph) + " User Query: " + query
        retrieved_docs = self.retrieve_documents(rag_query)[1]
        retrieved_docs.extend(self.retrieve_documents(query)[1])
        print(query)
        print(rag_query)
        print("Retrieved docs:", [d['metadata']['title'] for d in retrieved_docs], "Current docs:", self.current_graph)
        print("graph_list:", self.current_graph)

        current_docs = [self.title2doc[g] for g in self.current_graph if g in self.current_graph] + retrieved_docs
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

        print(graph_ops)
        graph = self.retrieve_graph(courses, certificates, self.current_graph)

        response=str(self.current_graph)
        self.past_graphs.append((query, self.current_graph))

        return response, graph

    def run_graph_rag_pipeline(self, query, courses, certificates):
        category = self.get_category(query)
        response = self.generate_response_based_on_category(query, category, courses, certificates)
        return response