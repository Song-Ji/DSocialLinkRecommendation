import networkx as nx
import linecache
import pandas as pd


def build_dynamic_network_from_txt(filename, start, end, network=None):
    # file_read = open(filename, 'r')
    if not network is None:
        nodes = set(network.nodes())
        edges = set(network.edges())
        g = network
    else:
        nodes = set()
        edges = set()
        g = nx.Graph(graphname=filename)

    print("start to build graph using txt file from {0} to {1}...".format(start, end))
    # build the graph from input file
    for i in range(start, end + 1):
        line_str = linecache.getline(filename, i)
        line = line_str.strip().strip('\n').split(' ')
        # print("{0}th line: {1}".format(i, line))
        node1 = int(line[0])
        node2 = int(line[1])
        timestamp = int(line[2])
        # add nodes to the graph
        g.add_nodes_from([node1, node2])
        nodes.update([node1, node2])
        # add edge to the graph, multiedge add to weight of the existing edge
        if g.has_edge(node1, node2):
            cur_weight = g[node1][node2]['weight']
            # g.edges[node1, node2]['weight'] = 1
            g[node1][node2]['weight'] = cur_weight + 1
        else:
            g.add_edge(node1, node2, weight=1)
        edges.add((node1, node2))
    print("graph building using txt file finished.")
    print("graph node size:{0}".format(g.number_of_nodes()))
    print("graph edge size:{0}".format(g.number_of_edges()))
    return g


def build_dynamic_network_from_dataframe(data: pd.DataFrame, start: int, end: int, network: nx.Graph = None):
    if not network is None:
        nodes = set(network.nodes())
        edges = set(network.edges())
        g = network
    else:
        nodes = set()
        edges = set()
        g = nx.Graph()

    print("start building graph using dataframe from {0} to {1}".format(start, end))
    data_for_building = data[(data['timestamp'] <= end) & (data['timestamp'] >= start)]  # 一定要用&号，用and会报错！！
    print(data_for_building)

    for i in range(0, len(data_for_building.index)):
        node1 = int(data_for_building.iloc[i, 0])
        node2 = int(data_for_building.iloc[i, 1])
        timestamp = int(data_for_building.iloc[i, 2])
        # add node
        g.add_nodes_from([node1, node2])
        nodes.update([node1, node2])
        # add edge to the graph, multiedge add to weight of the existing edge
        if g.has_edge(node1, node2):
            cur_weight = g[node1][node2]['weight']
            # g.edges[node1, node2]['weight'] = 1
            g[node1][node2]['weight'] = cur_weight + 1
        else:
            g.add_edge(node1, node2, weight=1)
        edges.add((node1, node2))
    print("graph building using dataframe finished.")
    print("graph node size:{0}".format(g.number_of_nodes()))
    print("graph edge size:{0}".format(g.number_of_edges()))
    return g


# edge_list should be a list of tuple (node1, node2, timestamp)
def add_edges_to_graph_from_list(graph: nx.Graph = None, edge_list:list = None):
    if graph is None:
        print("graph is None, error!")
        return
    if edge_list is None:
        print("edge_list is None, graph remain unchanged.")
        return graph

    g = graph
    for edge in edge_list:
        node1 = edge[0]
        node2 = edge[1]
        if g.has_edge(node1, node2):
            cur_weight = g[node1][node2]['weight']
            g[node1][node2]['weight'] = cur_weight + 1
        else:
            g.add_nodes_from([node1, node2])
            g.add_edge(node1, node2, weight=1)
    print("{0} edges added to graph-{1}.".format(len(edge_list), g.name))
    return g


def remove_edges_from_graph_from_list(graph: nx.Graph = None, edge_list: list = None):
    if graph is None:
        raise Exception("graph is None, error!")
    if edge_list is None:
        print("edge_list is None, graph remain unchanged.")
        return graph

    g = graph
    for edge in edge_list:
        node1 = edge[0]
        node2 = edge[1]
        if g.has_edge(node1, node2):
            cur_weight = g[node1][node2]['weight']
            if cur_weight > 1:
                g[node1][node2]['weight'] = cur_weight - 1
            else:
                g.remove_edge(node1, node2)
        else:
            raise Exception("try to remove an edge that does not exist!")
    print("{0} edges remove from graph-{1}.".format(len(edge_list), g.name))
    return g