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
from graph_utils import *
from rag import *

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

rag = BasicRAG(document_store=store)

@app.route('/api/survey', methods=['POST'])
def survey():
    data = request.get_json()
    print("Survey received:", data)
    return jsonify({"status": "ok", "received": data}), 200

@app.route('/api/get_graph', methods=['POST'])
def get_graph():
    data = request.get_json()
    message = data['category']
    
    response, category, graph = rag.run_rag_pipeline(message, courses, certificates, user_data)
    if len(graph) > 0:
        nodes, edges = graph_to_2d_array(graph)
    else:
        nodes, edges = [], []
    return jsonify({'nodes': nodes, 'edges': edges, 'message': response})
    # Perform any server-side actions here

if __name__ == '__main__':
    app.run(debug=True)