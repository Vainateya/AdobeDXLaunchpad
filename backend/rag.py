from openai import OpenAI
from documents import *
from graph_utils import *
from utils import *
import ast
import inspect
from enum import Enum
import re
import ast
from chat import Chatter
from pprint import pprint

class Bucket(Enum):
    IRRELEVANT = 1
    GENERAL = 2
    GRAPH = 3

COURSE_LEVELS = ['Foundations', 'Professional', 'Expert']
CERT_LEVELS = ['Professional', 'Expert', 'Master']

def yes_before_no(text: str):
    t = text.upper()
    i_yes = t.find("YES")
    i_no  = t.find("NO")

    if i_yes == -1 and i_no == -1:
        return None          
    if i_yes == -1:
        return False         
    if i_no == -1:
        return True         
    return i_yes < i_no

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
    
    def __init__(self, document_store: DocumentStore, supplement_store: DocumentStore, model: str = "gpt-4o-mini", temperature=0):
        self.document_store = document_store
        self.supplement_store = supplement_store
        self.model = model
        self.client = OpenAI()
        self.chat = Chatter(model, temperature)
        self.chat_history = [] 
        self.role = Bucket.IRRELEVANT
        self.past_graphs: tuple[str, list[str]] = [] # {user_query, [coursename, coursename, ...]}
        self.current_graph = [] # {}
        self.current_nx_graph = nx.DiGraph()
        self.title2doc = {d['metadata']['title']: d for d in self.document_store.get_all_documents()}
        self.temperature = 0

    def format_docs_context(self, docs):
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

    def format_past_graphs(self, graphs: list[str]):
        out = ''
        for i, (query, graph) in enumerate(graphs):
            out +=  f"At time {i}: User asked '{query}'. Resources in the resource graph at that time were {str(graph)}\n'"
        return out
    
    def retrieve_documents(self, query: str, top_k: int = 5, exclude_supplement: bool = False, omit_titles = []):
        """Fetches the top-k most relevant documents from the document store and formats metadata.
        If exclude_supplement=True, excludes supplemental docs (only uses course/cert content)."""

        all_docs = self.document_store.query_documents(query_text=query, top_k=top_k)
        filtered_docs = [doc for doc in all_docs if doc['metadata']['title'] not in omit_titles]

        if not exclude_supplement:
            supplement_docs = self.supplement_store.query_documents(query_text=query, top_k=top_k)
            filtered_docs += supplement_docs  # same as all_docs = all_docs + supplement_docs

        context = self.format_docs_context(filtered_docs)

        print("Similarity scores for query:", query)
        for doc in filtered_docs:
            print(doc['metadata']['title'], doc['score'])

        return context, filtered_docs

    def retrieve_graph(self, courses, certificates, sources):
        G = get_llm_graph(courses, certificates, sources)
        if len(G) == 0:
            graph = nx.DiGraph()
            return graph
        return G

    def add_to_history(self, role: str, content: str):
        """Appends a new entry to the chat history."""
        self.chat_history.append({"role": role, "content": content})

    def format_chat_history(self, history="") -> str:
        """Formats the chat history for including in the prompt."""
        if not history:
            history = self.chat_history
        
        chat_history_text = ""
        for entry in history:
            if entry["role"] == "user":
                chat_history_text += f"User previously said: {entry['content']}\n"
            elif entry["role"] == "assistant":
                chat_history_text += f"Assistant previously said: {entry['content']}\n"
        return chat_history_text

    def update_graph_state(self, query, courses, certificates, user_profile):
        bucket = self.find_bucket(query)
        if bucket == Bucket.IRRELEVANT:
            return self.current_nx_graph
    
        rag_query = 'Current resources in graph: ' + ', '.join("'" + name + "'" for name in self.current_graph) + " User Query: " + query

        retrieved_docs = self.retrieve_documents(rag_query, top_k=5, exclude_supplement=True)[1]
        retrieved_docs.extend(
            self.retrieve_documents(query, top_k=5, exclude_supplement=True, omit_titles=[d['metadata']['title'] for d in retrieved_docs])[1]
        )
        current_docs = [self.title2doc[g] for g in self.current_graph if g in self.current_graph and 'Study Materials' not in g] + retrieved_docs
        resource_info = self.format_docs_context(current_docs)
        print("retrieved_docs")
        pprint(retrieved_docs)

        raw_graph_ops = self.chat.generate_graph_call(
            query=query, 
            cur_resources=str(self.current_graph), 
            graph_history_text=self.format_past_graphs(self.past_graphs),
            resource_info=resource_info, 
            user_profile=user_profile, 
            chat_history_text=self.format_chat_history()
        )
        # raw_graph_ops = self.generate_new_graph(query, str(self.current_graph), self.format_past_graphs(self.past_graphs), resource_info, user_profile=user_profile)
        pprint(raw_graph_ops)
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

        self.add_to_history("user", query)
        if len(self.current_graph) > 0:
            _, e = graph_to_2d_array(self.current_nx_graph)
            graph_str = display_edges(e)
        else:
            graph_str = ""

        retrieved_docs = self.retrieve_documents(query, top_k)[1]
        raw_verification = self.chat.generate_hallucination_check_call(
            query=query, 
            content=self.format_docs_context(retrieved_docs) + self.format_chat_history()
        )
        print(raw_verification)
        passed_verification = yes_before_no(raw_verification)
        if not passed_verification:
            stream = self.chat.stream_deny_call(
                query=query,
                context=self.format_docs_context(retrieved_docs) + self.format_chat_history(),
                verification=passed_verification,
            )
        else:
            stream = self.chat.stream_general_response_call(
                query=query,
                documents=retrieved_docs,
                user_profile=user_profile,
                graph_str_raw=graph_str,
                chat_history_text=self.format_chat_history()
            )
            # stream = self.generate_general_response(query, retrieved_docs, user_profile, 'Graph' if bucket == Bucket.GRAPH else 'General', graph_str_raw=graph_str)

        full_response = ""
        for chunk in stream:
            full_response += chunk
            yield chunk

        # print(full_response)
        self.add_to_history("assistant", full_response)

    def find_bucket(self, query: str):
        if self.chat_history and query in [self.chat_history[-1]['content'], self.chat_history[-2]['content']]:
            return self.role
        
        response = self.chat.generate_grouper_call(query).strip()
        # response = self.grouper(query).strip()

        # if "1" in response:
        #     self.role = Bucket.GENERAL
        # elif "2" in response:
        #     self.role = Bucket.GRAPH

        if "1" in response:
            self.role = Bucket.GENERAL
        elif "2" in response:
            self.role = Bucket.GENERAL
        elif "3" in response:
            self.role = Bucket.GRAPH
        else:
            print("Warning: Unable to classify query response properly.")
            self.role = Bucket.IRRELEVANT  # fallback
        
        return self.role 
