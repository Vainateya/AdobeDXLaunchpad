from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from flask_caching import Cache
import os
import json
from tqdm import tqdm
from rag import SimpleAgenticRAG
import sys
sys.path.append('../dependency_graph')
from utils import Course, Certificate
from graph_utils import get_specific_graph, graph_to_2d_array


app = Flask(__name__)
CORS(app, supports_credentials=True)
app.config['CACHE_TYPE'] = 'simple'
cache = Cache(app)

rag_system = SimpleAgenticRAG()

# Load courses and certificates for graph functionality
courses = []
certificates = []

# Load courses from dependency_graph
course_numbers = [
        1043, 1045, 1046, 1047, 1048, 1049, 1050, 1054, 1055, 1056, 
        1057, 1058, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 
        1067, 1068, 1069, 1221
    ]
    
for n in tqdm(course_numbers, desc="Loading courses"):
    new_course = Course(f'https://certification.adobe.com/courses/{n}')
    courses.append(new_course)

# Load certificates from HTML files
certificate_htmls_location = '../dependency_graph/certificate_htmls'
if os.path.exists(certificate_htmls_location):
    for html in tqdm(os.listdir(certificate_htmls_location), desc="Loading certificates"):
        if html.endswith('.html'):
            try:
                certificate = Certificate(f'{certificate_htmls_location}/{html}')
                certificates.append(certificate)
            except Exception as e:
                print(f"Failed to load certificate {html}: {e}")
                continue

print(f"Loaded {len(courses)} courses and {len(certificates)} certificates")

@app.route('/api/survey', methods=['POST'])
def survey():
    data = request.get_json()
    print("Survey received:", data)
    rag_system.set_user_profile(data)
    return jsonify({"status": "ok", "received": data}), 200

@app.route('/api/update_graph', methods=['POST'])
def update_graph():
#     data = request.get_json()
#     query = data['category']
#     graph_enabled = data.get('graph_enabled', True)
#     if not graph_enabled:
#         return jsonify({
#             "nodes": [],
#             "edges": [],
#             "current_items": []
#         }), 200
    
#     graph = rag.update_graph_state(query, courses, certificates, user_data)

#     if len(graph) > 0:
#         nodes, edges = graph_to_2d_array(graph)
#         current_graph_items = [
#             node['display']
#             for row in nodes
#             for node in row
#             if 'display' in node
#         ]
#     else:
#         nodes, edges = [], []
#         current_graph_items = []
    nodes = []
    edges = []
    current_graph_items = []

    return jsonify({
        "nodes": nodes,
        "edges": edges,
        "current_items": current_graph_items
    }), 200

#     # Perform any server-side actions here

@app.route('/api/stream_response', methods=['POST'])
def stream_response():
    data = request.get_json()
    query = data['category']
    graph_enabled = data.get('graph_enabled', True)
    print("graph_enabled", graph_enabled)

    rag_system.add_message("user", query)
    response = rag_system.get_response(query)
    rag_system.add_message("assistant", response)

    def generate():
        rag_system.add_message("user", query)
        for token in rag_system.get_response(query):
            if token is not None:
                yield token.encode("utf-8")

    return Response(generate(), mimetype='text/plain')

@app.route('/api/get_graph_from_response', methods=['POST'])
def get_graph_from_response():
    """Generate a graph based on the last extracted courses/certificates from the RAG response."""
    try:
        extracted_items = rag_system.get_last_extracted_items()
        print("extracted_items", extracted_items)
        
        if not extracted_items or len(extracted_items) < 2:
            return jsonify({
                "nodes": [],
                "edges": [],
                "current_items": [],
                "message": "No courses/certificates found in the last response or not enough items for a meaningful graph."
            }), 200
        
        # Get the last user query from chat history
        user_query = None
        for msg in reversed(rag_system.chat_history):
            if msg.role == "user":
                user_query = msg.content
                break
        if not user_query:
            user_query = ""
        
        # Use AI to determine relevant_roles and info_level
        graph_params = rag_system.analyze_graph_parameters(extracted_items, user_query)
        relevant_roles = graph_params["relevant_roles"]
        info_level = graph_params["info_level"]
        resource_type = graph_params["resource_type"]
        print(f"AI-selected relevant_roles: {relevant_roles}, info_level: {info_level}, resource_type: {resource_type}")
        
        # Generate the graph using the extracted items as starting nodes
        graph = get_specific_graph(courses, certificates, relevant_roles, info_level, extracted_items, resource_type)
        
        if len(graph.nodes) == 0:
            return jsonify({
                "nodes": [],
                "edges": [],
                "current_items": [],
                "message": "No graph could be generated from the extracted items."
            }), 200
        
        # Convert graph to 2D array format for frontend
        nodes, edges = graph_to_2d_array(graph)
        
        # Flatten nodes for current_items
        current_graph_items = [
            node['display']
            for row in nodes
            for node in row
            if 'display' in node
        ]
        
        return jsonify({
            "nodes": nodes,
            "edges": edges,
            "current_items": current_graph_items,
            "extracted_items": extracted_items,
            "relevant_roles": relevant_roles,
            "info_level": info_level,
            "resource_type": resource_type,
            "message": f"Generated graph with {len(current_graph_items)} items from {len(extracted_items)} extracted courses/certificates."
        }), 200
        
    except Exception as e:
        print(f"Error generating graph: {e}")
        return jsonify({
            "nodes": [],
            "edges": [],
            "current_items": [],
            "message": f"Error generating graph: {str(e)}"
        }), 500


@app.route('/api/set_current_graph', methods=['POST'])
def set_current_graph():
    # data = request.get_json()
    # nodes_nested = data.get('nodes', [])
    # nodes = [node for sublist in nodes_nested for node in sublist]
    # print("Raw nodes from frontend:", nodes)
    # current_items = [node['data']['display'] for node in nodes if 'data' in node and 'display' in node['data']]
    # print("Updated Current Graph Items from frontend:", current_items)

    # rag.current_graph = current_items
    # rag.current_nx_graph = get_llm_graph(courses, certificates, current_items)
    # rag.past_graphs.append(("I want to go back to a previous graph.", current_items))
    current_items = []

    return jsonify({"status": "ok", "current_items": current_items}), 200

@app.route('/api/get_extracted_items', methods=['GET'])
def get_extracted_items():
    """Get the last extracted courses/certificates from the RAG response."""
    extracted_items = rag_system.get_last_extracted_items()
    return jsonify({
        "extracted_items": extracted_items,
        "count": len(extracted_items)
    }), 200

if __name__ == '__main__':
    app.run(debug=True)