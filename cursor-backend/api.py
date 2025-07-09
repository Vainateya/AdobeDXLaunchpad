from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from flask_caching import Cache
import os
import json
from tqdm import tqdm
from rag import SimpleAgenticRAG


app = Flask(__name__)
CORS(app, supports_credentials=True)
app.config['CACHE_TYPE'] = 'simple'
cache = Cache(app)

rag_system = SimpleAgenticRAG()

@app.route('/api/survey', methods=['POST'])
def survey():
    global user_data
    data = request.get_json()
    print("Survey received:", data)
    user_data = data
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
    # graph_enabled = data.get('graph_enabled', True)
    # print("graph_enabled", graph_enabled)

    rag_system.add_message("user", query)
    response = rag_system.get_response(query)
    rag_system.add_message("assistant", response)

    def generate():
        # rag.graph_enabled = graph_enabled
        rag_system.add_message("user", query)
        # for token in rag.run_rag_pipeline_stream(query, courses, certificates, user_data):
        #     if token is not None:
        #         yield token.encode("utf-8")
        for token in rag_system.get_response(query):
            if token is not None:
                yield token.encode("utf-8")

    return Response(generate(), mimetype='text/plain')


@app.route('/api/set_current_graph', methods=['POST'])
def set_current_graph():
    # data = request.get_json()
    # nodes_nested = data.get('nodes', [])
    # nodes = [node for sublist in nodes_nested for node in sublist]
    # print("📦 Raw nodes from frontend:", nodes)
    # current_items = [node['data']['display'] for node in nodes if 'data' in node and 'display' in node['data']]
    # print("📌 Updated Current Graph Items from frontend:", current_items)

    # rag.current_graph = current_items
    # rag.current_nx_graph = get_llm_graph(courses, certificates, current_items)
    # rag.past_graphs.append(("I want to go back to a previous graph.", current_items))
    current_items = []

    return jsonify({"status": "ok", "current_items": current_items}), 200

if __name__ == '__main__':
    app.run(debug=True)