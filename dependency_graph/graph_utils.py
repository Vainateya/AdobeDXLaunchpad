import networkx as nx
from utils import *

def create_graph(course_name, courses, certificates):
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

def flatten(l):
    return [
        src
        for row in l
        for src in row 
    ]

def display_nodes(nodes):
    res = 'The relevant sources (nodes) are as follows: \n'
    for src in flatten(nodes):
        res += f'{src["type"]} - {src["display"]} - {src["description"]}\n'
    return res

def display_edges(edges):
    res = 'These nodes are connected via a directed edge as follows: \n'
    for edge in edges:
        res += f'From: {edge["from"]}, To: {edge["to"]}\n'
    return res

def get_full_string(nodes, edges):
    return display_nodes(nodes) + display_edges(edges)