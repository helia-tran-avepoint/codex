from nebula3.Config import Config
from nebula3.data.ResultSet import ResultSet
from nebula3.gclient.net import ConnectionPool
from streamlit_agraph import Edge, Node

from webui import app_config


def connect_to_nebula():
    config = Config()
    config.max_connection_pool_size = 10
    connection_pool = ConnectionPool()

    host, port = app_config.nebula_address.split(":")
    if not connection_pool.init([(host, int(port))], config):
        raise Exception("Nebula connection failed!")
    return connection_pool


def get_graph_from_nebula(connection_pool, current_node=None):
    session = connection_pool.get_session(
        app_config.nebula_user, app_config.nebula_password
    )
    switch_space = session.execute(f"USE {app_config.space_name}")
    if not switch_space.is_succeeded():
        raise Exception(
            f"Failed to switch to space {app_config.space_name}: {switch_space.error_msg()}"
        )

    if current_node:
        query = f"""
        MATCH (n {{vid: "{current_node}"}})-[r]->(m)
        RETURN n, r, m LIMIT 50;
        """
    else:
        query = """
        MATCH (n)-[r]->(m)
        RETURN n, r, m LIMIT 100;
        """
    result = session.execute(query)
    if not result.is_succeeded():
        raise Exception(f"Query failed: {result.error_msg()}")

    nodes = {}
    edges = []

    for row in result:
        src = row["n"].as_node()
        dst = row["m"].as_node()
        edge = row["r"].as_relationship()

        nodes[src.get_id()] = Node(id=src.get_id(), label=src.get_id())
        nodes[dst.get_id()] = Node(id=dst.get_id(), label=dst.get_id())

        edges.append(
            Edge(
                source=src.get_id(),
                target=dst.get_id(),
                label=edge.edge_name,
            )
        )

    session.release()
    return list(nodes.values()), edges
