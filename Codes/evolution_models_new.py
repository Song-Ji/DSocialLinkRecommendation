import networkx as nx
import math
import calculate_param

weighted_model_num = 2
unweighted_model_num = 0

weighted_model_map = {0: "L_P_WCN",
                      1: "L_P_WAA"}
unweighted_model_map = {}

def L_P_CN(network: nx.Graph):
    num_add = 0  # the number of egdes to be added
    nodes_pair_without_edge = []  # the pairs of nodes without edges
    probability_add = []  # the probabilities of the pairs of nodes to be added
    score = 0  # the score of each pair of nodes in link prediction model
    total_score_without_edge = 0.0  # the sum of scores of pairs of nodes without edge

    #  calculate the score of each pair of nodes
    for i, elei in enumerate(list(network.nodes()), 1):
        for j, elej in enumerate(list(network.nodes()), 1):
            score = 0
            if i >= j:
                continue
            if not network.has_edge(elei, elej):
                try:
                    score = len(nx.common_neighbors(network, elei, elej))
                except:
                    continue
                total_score_without_edge += score
                nodes_pair_without_edge.append((elei, elej, score))

    for a, b, c in nodes_pair_without_edge:
        probability_add.append(c / total_score_without_edge)  # calculate the probabilities of edges to be added

    return nodes_pair_without_edge, probability_add


def L_P_WCN(network: nx.Graph):
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

    return nodes_pair, probability_add


def L_P_AA(network: nx.Graph):
    num_add = 0  # the number of egdes to be added
    nodes_pair_without_edge = []  # the pairs of nodes without edges
    probability_add = []  # the probabilities of the pairs of nodes to be added
    score = 0.0  # the score of each pair of nodes in link prediction model
    total_score_without_edge = 0.0  # the sum of scores of pairs of nodes without edge

    #  calculate the score of each pair of nodes
    for i, elei in enumerate(list(network.nodes(), 1)):
        for j, elej in enumerate(list(network.nodes(), 1)):
            score = 0.0
            if i >= j:
                continue
            if not network.has_edge(elei, elej):
                try:
                    pre = nx.admic_adar_index(network, [(elei, elej)])
                    for u, v, s in pre:
                        score = s
                except:
                    continue
                total_score_without_edge += score
                nodes_pair_without_edge.append((elei, elej, score))

    for a, b, c in nodes_pair_without_edge:
        probability_add.append(c / total_score_without_edge)  # calculate the probabilities of edges to be added

    return nodes_pair_without_edge, probability_add


def L_P_WAA(network):
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

    return nodes_pair, probability_add

def L_P_WJC(network):
    num_add = 0  # the number of egdes to be added
    nodes_pair = []  # the pairs of nodes with edges and without edges
    probability_add = []  # the probabilities of the pairs of nodes to be added
    score = 0.0  # the score of each pair of nodes in link prediction model
    total_score = 0.0  # the sum of scores of pairs of nodes with edge and without edge

    #  calculate the score of each pair of nodes
    for i, elei in enumerate(list(network.nodes()), 1):
        for j, elej in enumerate(list(network.nodes()), 1):
            score = 0.0
            if i >= j:
                continue
            try:
                list_cm = nx.common_neighbors(network, elei, elej)
                total_w_elei_z_elej = 0
                total_w_z_elei = 0
                total_w_z_elej = 0
                total_min_w = 0

                for z in list_cm:
                    w_elei_z = network.get_edge_data(elei, z).get('weight')
                    w_z_elej = network.get_edge_data(z, elej).get('weight')
                    w_elei_z_elej = w_elei_z + w_z_elej
                    total_w_elei_z_elej += w_elei_z_elej

                for z in network.neighbors(elei):
                    w_z_elei = network.get_edge_data(z, elei).get('weight')
                    total_w_z_elei += w_z_elei

                for z in network.neighbors(elej):
                    w_z_elej = network.get_edge_data(z, elej).get('weight')
                    total_w_z_elej += w_z_elej

                for z in list_cm:
                    w_elei_z = network.get_edge_data(elei, z).get('weight')
                    w_z_elej = network.get_edge_data(z, elej).get('weight')
                    min_w = min(w_elei_z, w_z_elej)
                    total_min_w += min_w

                score = total_w_1_2 / total_w_x_elei + total_w_y_elej - total_min_w
            except:
                continue
            total_score += score
            nodes_pair.append((elei, elej, score))

    for a, b, c in nodes_pair:
        probability_add.append(c / total_score)  # calculate the probabilities of edges to be added

    return nodes_pair, probability_add
