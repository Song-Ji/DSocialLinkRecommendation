import networkx as nx
import math
import calculate_param

weighted_model_num = 2
unweighted_model_num = 0

weighted_model_map = {0: "L_P_WCN",
                      1: "L_P_WAA"}
unweighted_model_map = {}


def L_P_WCN(network: nx.Graph, num_add):
    nodes_pair = []  # the pairs of nodes with edges and without edges
    probability_add = []  # the probabilities of the pairs of nodes to be added
    score = 0.0  # the score of each pair of nodes in link prediction model
    total_score = 0.0  # the sum of scores of pairs of nodes without edge and with edge

    # calculate the score of each pair of nodes
    for i, elei in enumerate(list(network.nodes()), 1):
        for j, elej in enumerate(list(network.nodes()), 1):
            # initialize score for each edge
            score = 0.0
            if i >= j:
                continue
            try:
                for z in nx.common_neighbors(network, elei, elej):
                    w_elei_z = network.get_edge_data(elei, z).get('weight')
                    w_z_elej = network.get_edge_data(z, elej).get('weight')
                    score += w_elei_z + w_z_elej
            except:
                continue
            total_score += score
            nodes_pair.append((elei, elej, score))

    for a, b, c in nodes_pair:
        probability_add.append(c / total_score)  # calculate the probabilities of edges to be added

    # select edges to be added according to probabilities
    edges_add = calculate_param.prob_select(nodes_pair, probability_add, num_add)
    '''
    for a, b, c in edges_add:
        network.add_edge(a, b)  # add selected edges
    '''
    return edges_add


def L_P_WAA(network, num_add):
    nodes_pair = []  # the pairs of nodes with edges and without edges
    probability_add = []  # the probabilities of the pairs of nodes to be added
    score = 0.0  # the score of each pair of nodes in link prediction model
    total_score = 0.0  # the sum of scores of pairs of nodes with edge and without edge

    #  calculate the score of each pair of nodes
    for i, elei in enumerate(list(network.nodes()), 1):
        for j, elej in enumerate(list(network.nodes()), 1):
            # initialize score for each edge
            score = 0.0
            if i >= j:
                continue
            try:
                list_cm = nx.common_neighbors(network, elei, elej)
                total_w_x_z = 0

                for z in list_cm:
                    w_elei_z = network.get_edge_data(elei, z).get('weight')
                    w_z_elej = network.get_edge_data(z, elej).get('weight')
                    w_elei_z_elej = w_elei_z + w_z_elej
                    for x in network.neighbors(z):
                        w_x_z = network.get_edge_data(x, z).get('weight')
                        total_w_x_z += w_x_z
                    score += w_elei_z_elej / math.log(1 + total_w_x_z)
            except:
                continue
            total_score += score
            nodes_pair.append((elei, elej, score))

    for a, b, c in nodes_pair:
        probability_add.append(c / total_score)  # calculate the probabilities of edges to be added

    # select edges to be added according to probabilities
    edges_add = calculate_param.prob_select(nodes_pair, probability_add, num_add)
    '''
    for a, b, c in edges_add:
        network.add_edge(a, b)  # add selected edges
    '''
    return edges_add
