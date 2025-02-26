from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_caching import Cache

from langchain_community.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import os
import streamlit as st
import graphviz as graphviz

from utils import *

# embedding_model = HuggingFaceEmbeddings(
#     model_name="thenlper/gte-small",
#     multi_process=True,
#     model_kwargs={"device": "cpu"},
#     encode_kwargs={"normalize_embeddings": True},
# )

courses = []
certificates = []

course_numbers = [
    1043, 1045, 1046, 1047, 1048, 1049, 1050, 1054, 1055, 1056, 
    1057, 1058, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 
    1067, 1068, 1069, 1221
]

for n in tqdm(course_numbers):
    new_course = Course(f'https://certification.adobe.com/courses/{n}')
    courses.append(new_course)

print(len(courses))

certificate_htmls_location = 'certificate_htmls'

for html in tqdm(os.listdir(certificate_htmls_location)):
    certificate = Certificate(f'{certificate_htmls_location}/{html}')
    certificates.append(certificate)

app = Flask(__name__)
CORS(app)
app.config['CACHE_TYPE'] = 'simple'
cache = Cache(app)

def create_graph(course_name):
    H = nx.DiGraph()

    sources = [node for node in courses + certificates if node.category == course_name]

    for src in sources:
        H.add_node(src)

    for i, src1 in enumerate(sources):
        for j, src2 in enumerate(sources):
            if i == j:
                continue
            if src1.is_prereq_to(src2):
                H.add_edge(src1, src2, color='black')
    return H

def graph_to_2d_array(G):
    root_nodes = [n for n in G.nodes if G.in_degree(n) == 0]

    row_map = {}
    queue = [(node, 0) for node in root_nodes]

    while queue:
        node, row = queue.pop(0)

        if node in row_map:
            row = max(row, row_map[node])
        row_map[node] = row

        for child in G.successors(node):
            if child in row_map:
                row_map[child] = max(row_map[child], row + 1)
            else:
                row_map[child] = row + 1
            queue.append((child, row_map[child]))

    max_row = max(row_map.values())
    graph_2d = [[] for _ in range(max_row + 1)]

    for node, row in row_map.items():
        if type(node) == Course:
            node_type = 'course'
            desc = f'Course objectives: {node.objectives}'
        else:
            node_type = 'certificate'
            desc = f'Prerequisites: {node.prereq}'
        graph_2d[row].append({'type': node_type, 'display': node.display, 'description': desc})

    edges_2d = []
    for u, v in G.edges():
        if row_map[u] < row_map[v]:  # Ensure edges go downward
            edges_2d.append({'from': u.display, 'to': v.display})

    return graph_2d, edges_2d

@app.route('/api/get_graph', methods=['POST'])
def get_graph():
    data = request.get_json()
    course_name = data['category']
    G = create_graph(course_name)
    nodes, edges = graph_to_2d_array(G)
    return jsonify({'nodes': nodes, 'edges': edges, 'message': 'Graph displayed'})
    # Perform any server-side actions here

if __name__ == '__main__':
    app.run(debug=True)