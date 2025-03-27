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

course_numbers = [
    1043, 1045, 1046, 1047, 1048, 1049, 1050, 1054, 1055, 1056, 
    1057, 1058, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 
    1067, 1068, 1069, 1221
]

courses = []
course_embeddings = []

for n in tqdm(course_numbers):
    new_course = Course(f'https://certification.adobe.com/courses/{n}')
    courses.append(new_course)

print(len(courses))

certificate_htmls_location = 'certificate_htmls'
certificates = []
certificate_embeddings = []

for html in tqdm(os.listdir(certificate_htmls_location)):
    certificate = Certificate(f'{certificate_htmls_location}/{html}')
    certificates.append(certificate)

course_name = 'Adobe Analytics'

def create_graph(course_name):

    sources = [node for node in courses + certificates if node.category == course_name]

    nodes = []
    for i, src in enumerate(sources):
        if type(src) == Course:
            nodes.append({
                'id': i,
                'label': src.display,
                'color': 'blue'
            })
        else:
            nodes.append({
                'id': i,
                'label': src.display,
                'color': 'green'
            })

    edges = []

    for i, src1 in enumerate(sources):
        for j, src2 in enumerate(sources):
            if i == j:
                continue
            if src1.is_prereq_to(src2):
                edges.append(
                    {'from': i, 'to': j, 'color': 'green'}
                )
            elif src1.has_same(src2, 'category'):
                edges.append(
                    {'from': i, 'to': j, 'color': 'red'}
                )

    return nodes, edges

nodes, edges = create_graph('Adobe Analytics')
