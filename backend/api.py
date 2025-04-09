from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_caching import Cache
import os

from tqdm import tqdm
import os

from utils import *
from graph_utils import *
from rag import *

# embedding_model = HuggingFaceEmbeddings(
#     model_name="thenlper/gte-small",
#     multi_process=True,
#     model_kwargs={"device": "cpu"},
#     encode_kwargs={"normalize_embeddings": True},
# )

# Create separate DocumentStore for program info documents
program_store = DocumentStore(similarity_metric="cosine", storage_path="./supplement_docs")

# Add .txt documents to the program_store
for fname in os.listdir("./supplemental_sources"):
    if fname.endswith(".txt"):
        doc = TextDocument(os.path.join("./supplemental_sources", fname))
        program_store.add_document(doc)
        

courses = []
certificates = []

course_numbers = [
    1043, 1045, 1046, 1047, 1048, 1049, 1050, 1054, 1055, 1056, 
    1057, 1058, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 
    1067, 1068, 1069, 1221
]

certificate_htmls_location = '../dependency_graph/certificate_htmls'

save_folder = "pickels"
certificates_save_location = os.path.join(save_folder, "certificates.pkl")
courses_save_location = os.path.join(save_folder, "courses.pkl")

user_data = {}

if not os.path.exists(courses_save_location) or not os.path.exists(certificates_save_location):
    print("Scraping resources...")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    for n in tqdm(course_numbers):
        new_course = Course(f'https://certification.adobe.com/courses/{n}')
        courses.append(new_course)
    save_sources_pickle(courses, courses_save_location)

    for html in tqdm(os.listdir(certificate_htmls_location)):
        certificate = Certificate(f'{certificate_htmls_location}/{html}')
        certificates.append(certificate)
    save_sources_pickle(certificates, certificates_save_location)
else:
    print("Loading resources...")
    certificates = load_sources_pickle(certificates_save_location)
    courses = load_sources_pickle(courses_save_location)


app = Flask(__name__)
CORS(app, supports_credentials=True)
app.config['CACHE_TYPE'] = 'simple'
cache = Cache(app)

store = DocumentStore(similarity_metric="cosine", storage_path="./chroma_storage")
if store.collection.count() == 0:
    print("No documents found in ChromaDB. Re-adding documents...")
    for course in courses:
        store.add_document(course)
    for certificate in certificates:
        store.add_document(certificate)
    print(f"Total documents after reloading: {store.collection.count()}")

rag = BasicRAG(document_store=store, supplement_store = program_store)

@app.route('/api/survey', methods=['POST'])
def survey():
    global user_data
    data = request.get_json()
    print("Survey received:", data)
    user_data = data
    return jsonify({"status": "ok", "received": data}), 200

@app.route('/api/get_graph', methods=['POST'])
def get_graph():
    global current_graph_items

    data = request.get_json()
    message = data['category']
    
    response, graph = rag.run_rag_pipeline(message, courses, certificates, user_data)

    if len(graph) > 0:
        nodes, edges = graph_to_2d_array(graph)

        # Flatten and extract node names
        current_graph_items = [
            node['display']
            for row in nodes
            for node in row
            if 'display' in node
        ]
    else:
        nodes, edges = [], []
        current_graph_items = []

    return jsonify({
        "nodes": nodes,
        "edges": edges,
        "message": response,
        "current_items": current_graph_items
    }), 200


    # Perform any server-side actions here

@app.route('/api/set_current_graph', methods=['POST'])
def set_current_graph():
    data = request.get_json()
    nodes_nested = data.get('nodes', [])
    nodes = [node for sublist in nodes_nested for node in sublist]
    print("📦 Raw nodes from frontend:", nodes)
    current_items = [node['data']['display'] for node in nodes if 'data' in node and 'display' in node['data']]
    print("📌 Updated Current Graph Items from frontend:", current_items)

    rag.current_graph = current_items
    rag.current_nx_graph = get_llm_graph(courses, certificates, current_items)
    rag.past_graphs.append(("I want to go back to a previous graph.", current_items))

    return jsonify({"status": "ok", "current_items": current_items}), 200

if __name__ == '__main__':
    app.run(debug=True)