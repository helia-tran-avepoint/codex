import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import community.community_louvain as community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from community import community_louvain

from index_service import app_config
from index_service.custom_store import GraphRAGStore

graph_store = GraphRAGStore(
    username=app_config.graph_db_username,
    password=app_config.graph_db_password,
    url=app_config.graph_db_url,
    database=app_config.graph_db_database_name,
    refresh_schema=True,
    sanitize_query_output=False,
)


G = graph_store._create_nx_graph()


def find_all_call_chains(graph):
    call_chains = []
    start_nodes = [node for node in graph.nodes if graph.in_degree(node) == 0]

    end_nodes = [node for node in graph.nodes if graph.out_degree(node) == 0]

    for start in start_nodes:
        for end in end_nodes:
            for path in nx.all_simple_paths(graph, source=start, target=end):
                call_chains.append(path)
    return call_chains


call_chains = find_all_call_chains(G)

print("\nBiz Call Chain:")
for i, chain in enumerate(call_chains):
    print(f"Method chain: {i+1}: {' -> '.join(chain)}")

plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G)

for i, path in enumerate(call_chains):
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="lightblue",
        edge_color="gray",
        alpha=0.7,
        font_size=10,
    )
    edges_in_path = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
    nx.draw_networkx_edges(G, pos, edgelist=edges_in_path, edge_color="red", width=2)

plt.savefig("method_call_chains.png", dpi=300, format="png")
plt.show()
