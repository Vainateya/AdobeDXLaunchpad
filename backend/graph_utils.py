import networkx as nx
from collections import deque
from utils import *

# def create_graph(course_name, courses, certificates):
#     H = nx.DiGraph()

#     sources = [node for node in courses + certificates if node.category == course_name]

#     for src in sources:
#         H.add_node(src)

#     for i, src1 in enumerate(sources):
#         for j, src2 in enumerate(sources):
#             if i == j:
#                 continue
#             if src1.is_prereq_to(src2):
#                 H.add_edge(src1, src2, color='black')
#     return H

# Example parameters
# 
# relevant_roles = ['All', 'Business Practitioner']
# info_level = 'medium' 
# starting_node = {
#     'category': 'Adobe Commerce',
#     'level': 'Foundations',
#     'type': Course
# }
# job_role_query = "I want to primarily work with digital marketing, such as advertising."

sample = ['Adobe Analytics Foundations', 'Adobe Analytics Architect Master', 'Adobe Analytics Business Practitioner Expert', 'Adobe Analytics Business Practitioner Professional', 'Adobe Analytics Developer Expert', 'Adobe Analytics Developer Professional']

def get_llm_graph(courses, certificates, sources=sample):
    """
    Create a directed graph based on a list of relevant sources given by a llm call
    """

    G = nx.DiGraph()
    relevant = []
    for i in courses + certificates:
        if i.display in sources:
            relevant.append(i)
            G.add_node(i)
            if type(i) == Certificate:
                G.add_node(i.study)
                G.add_edge(i.study, i)
    
    for src1 in relevant:
        for src2 in relevant:
            if src1 == src2:
                continue
            if src1.is_prereq_to(src2):
                if type(src2) == Certificate:
                    G.add_edge(src1, src2.study)
                else:
                    G.add_edge(src1, src2)
    
    return G
    


def get_specific_graph(courses, certificates, relevant_roles, info_level, starting_nodes):
    """
    Create a directed graph of courses and certificates based on prerequisite relationships.

    Parameters:
        courses (list): List of course objects.
        certificates (list): List of certificate objects.
        relevant_roles (list): List of job roles to filter the courses and certificates.
        info_level (str): Level of graph expansion ('low', 'medium', or 'high').
        starting_node (dict): Dictionary with keys 'category', 'level', and 'type' specifying the starting node.

    Returns:
        networkx.DiGraph: A directed graph representing the prerequisite relationships.
    """

    G = nx.DiGraph()
    queue = deque()
    sources = []
    for i in courses + certificates:
        if i.job_role == 'All':
            sources.append(i)
        elif i.job_role in relevant_roles:
            sources.append(i)
        else:
            if 'All' in relevant_roles:
                sources.append(i)

    # first, we find the root node(s)

    for src in sources:
        if src.display in starting_nodes and src not in G:
            queue.append(src)
            G.add_node(src)
            if type(src) == Certificate:
                G.add_node(src.study)
                G.add_edge(src.study, src)

            if info_level == 'low':
                pass

            if info_level == 'medium':
                certificate_in_graph = False
                while queue and not certificate_in_graph:
                    node = queue.popleft()

                    for i, src in enumerate(sources):
                        if node.is_prereq_to(src):
                            if src not in G:
                                G.add_node(src)
                                queue.append(src)
                                if type(src) == Certificate and src.display not in starting_nodes:
                                    certificate_in_graph = True
                                if type(src) == Certificate:
                                    G.add_node(src.study)
                                    G.add_edge(src.study, src)
                                    src = src.study
                            G.add_edge(node, src)

            if info_level == 'high':
                while queue:
                    node = queue.popleft()

                    for i, src in enumerate(sources):
                        if node.is_prereq_to(src):
                            if src not in G:
                                G.add_node(src)
                                queue.append(src)
                                if type(src) == Certificate:
                                    G.add_node(src.study)
                                    G.add_edge(src.study, src)
                            if type(src) == Certificate:
                                G.add_edge(node, src.study)
                            else:
                                G.add_edge(node, src)

    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True, font_size=6, node_size=40)
    return G

def graph_to_2d_array(G):
    if len(G.nodes) == 0:
        return [], []
    
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
        elif type(node) == Certificate:
            node_type = 'certificate'
        else:
            node_type = 'study'
        graph_2d[row].append({'type': node_type, 'display': node.display, 'data': node.to_dict()})

    edges_2d = []
    for u, v in G.edges():
        if row_map[u] < row_map[v]:  # Ensure edges go downward
            edges_2d.append({'from': u.display, 'to': v.display})

    return graph_2d, edges_2d

def flatten(l):
    return [
        src
        for row in l
        for src in row 
    ]

def display_nodes(nodes):
    res = 'The relevant sources (nodes) are as follows: \n'
    for src in flatten(nodes):
        res += f'{src["type"]} - {src["display"]} - {src["data"]}\n'
    return res

def display_edges(edges):
    res = 'These nodes are connected via a directed edge as follows: \n'
    for edge in edges:
        res += f'From: {edge["from"]}, To: {edge["to"]}\n'
    return res

def get_full_string(nodes, edges):
    return display_nodes(nodes) + display_edges(edges)

def graph_to_json(nodes, edges):
    flat_nodes = list(flatten(nodes))
    return {
        "nodes": [
            {
                "type": node["type"],
                "name": node["display"],
                "data": node["data"]
            }
            for node in flat_nodes
        ],
        "edges": [
            {
                "from": edge["from"],
                "to": edge["to"]
            }
            for edge in edges
        ]
    }

def get_graph_json_string(nodes, edges, indent=2):
    """Return a pretty-printed JSON string version of the graph"""
    return json.dumps(graph_to_json(nodes, edges), indent=indent)