import networkx as nx
import math
import random
import calculate_param
import numpy as np
import sklearn as sl
from collections import defaultdict
import copy


def L_P_CN(network):
    num_add = 0  # the number of egdes to be added
    nodes_pair_without_edge = []  # the pairs of nodes without edges
    probability_add = []  # the probabilities of the pairs of nodes to be added
    score = 0  # the score of each pair of nodes in link prediction model
    total_score_without_edge = 0.0  # the sum of scores of pairs of nodes without edge

    #  calculate the score of each pair of nodes
    for i, elei in enumerate(list(network.nodes()), 1):
        for j, elej in enumerate(list(network.nodes()), 1):

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
    # select edges to be added according to probabilities
    edges_add = calculate_param.prob_select_distinct(nodes_pair_without_edge, probability_add, num_add)
    for a, b, c in edges_add:
        network.add_edge(a, b)  # add selected edges

    return True


def L_P_WCN(network):
    num_add = 0  # the number of egdes to be added
    nodes_pair = []  # the pairs of nodes with edges and without edges
    probability_add = []  # the probabilities of the pairs of nodes to be added
    score = 0.0  # the score of each pair of nodes in link prediction model
    total_score = 0.0  # the sum of scores of pairs of nodes without edge and with edge

    #  calculate the score of each pair of nodes
    for i, elei in enumerate(list(network.nodes()), 1):
        for j, elej in enumerate(list(network.nodes()), 1):

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

    for a, b, c in edges_add:
        network.add_edge(a, b)  # add selected edges

    return True


def L_P_RWWR(network):
    num_add = 0  # the number of egdes to be added
    nodes_pair_without_edge = []  # the pairs of nodes without edges
    probability_add = []  # the probabilities of the pairs of nodes to be added
    score = 0  # the score of each pair of nodes in link prediction model
    total_score_without_edge = 0.0  # the sum of scores of pairs of nodes without edge

    #  calculate the score of each pair of nodes
    for i, elei in enumerate(list(network.nodes()), 1):
        for j, elej in enumerate(list(network.nodes()), 1):

            if i >= j:
                continue
            if not network.has_edge(elei, elej):
                try:
                    score = (CCD(network, elei, elej, i, j, 0.15) + CCD(network, elej, elei, j, i, 0.15)) / 2
                except:
                    continue
                total_score_without_edge += score
                nodes_pair_without_edge.append((elei, elej, score))

    for a, b, c in nodes_pair_without_edge:
        probability_add.append(c / total_score_without_edge)  # calculate the probabilities of edges to be added
    # select edges to be added according to probabilities
    edges_add = calculate_param.prob_select_distinct(nodes_pair_without_edge, probability_add, num_add)
    for a, b, c in edges_add:
        network.add_edge(a, b)  # add selected edges

    return True


def L_P_WRWWR(network):
    num_add = 0  # the number of egdes to be added
    nodes_pair = []  # the pairs of nodes with edges and without edges
    probability_add = []  # the probabilities of the pairs of nodes to be added
    score = 0.0  # the score of each pair of nodes in link prediction model
    total_score = 0.0  # the sum of scores of pairs of nodes with edge and without edge

    #  calculate the score of each pair of nodes
    for i, elei in enumerate(list(network.nodes()), 1):
        for j, elej in enumerate(list(network.nodes()), 1):

            if i >= j:
                continue
            try:
                score = (CCD(network, elei, elej, i, j, 0.15) + CCD(network, elej, elei, j, i, 0.15)) / 2
            except:
                continue
            total_score += score
            nodes_pair.append((elei, elej, score))

    for a, b, c in nodes_pair:
        probability_add.append(c / total_score)  # calculate the probabilities of edges to be added
    # select edges to be added according to probabilities
    edges_add = calculate_param.prob_select(nodes_pair, probability_add, num_add)
    for a, b, c in edges_add:
        network.add_edge(a, b)  # add selected edges

    return True


# Algorithm-CCD: Cross-modal correlation discovery for Random Walk with Restart
def CCD(network, node_i, node_j, i, j, restart_prob):
    v = np.array([len(network.nodes()) * [0]]).T  # The v vector is a column vector with all its N elements zero,
    v[i - 1] = 1  # except for the entry that corresponds to starting nodeself and set it as '1'
    A = nx.to_numpy_array(network)  # A is the adjacency matrix of the network
    A_column_normalized = sl.normalize(A, "l1", 0)
    u = copy.deepcopy(
        v)  # u_q is the steady state probability vector = (u_q(1),....u_q(N)). N is the number of nodes in the network ; Initialize u = v
    u_old = []  # the u vector before converged
    max_iter = 100  # the maximum iteration times

    for x in range(max_iter):
        u_old = copy.deepcopy(u)
        u = (1 - restart_prob) * np.dot(A_column_normalized, u) + np.dot(restart_prob, v)
        if _is_vector_converge(u, u_old):
            break

    return u[j - 1]


def _is_vector_converge(u, u_old):
    if abs(u[0] - u_old[0]) >= 1e-4:
        return False
    return True


def L_P_JC(network):
    num_add = 0  # the number of egdes to be added
    nodes_pair_without_edge = []  # the pairs of nodes without edges
    probability_add = []  # the probabilities of the pairs of nodes to be added
    score = 0  # the score of each pair of nodes in link prediction model
    total_score_without_edge = 0.0  # the sum of scores of pairs of nodes without edge

    #  calculate the score of each pair of nodes
    for i, elei in enumerate(list(network.nodes(), 1)):
        for j, elej in enumerate(list(network.nodes(), 1)):

            if i >= j:
                continue
            if not network.has_edge(elei, elej):
                try:
                    pre = nx.jaccard_coefficient(network, [(elei, elej)])
                    for u, v, s in pre:
                        score = s
                except:
                    continue
                total_score_without_edge += score
                nodes_pair_without_edge.append((elei, elej, score))

    for a, b, c in nodes_pair_without_edge:
        probability_add.append(c / total_score_without_edge)  # calculate the probabilities of edges to be added
    # select edges to be added according to probabilities
    edges_add = calculate_param.prob_select_distinct(nodes_pair_without_edge, probability_add, num_add)
    for a, b, c in edges_add:
        network.add_edge(a, b)  # add selected edges

    return True


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
    # select edges to be added according to probabilities
    edges_add = calculate_param.prob_select(nodes_pair, probability_add, num_add)
    for a, b, c in edges_add:
        if (network.has_edge(a, b)):
            network[a][b]['weight'] += 1
        else:
            network.add_edge(a, b)

    return True


def L_P_AA(network):
    num_add = 0  # the number of egdes to be added
    nodes_pair_without_edge = []  # the pairs of nodes without edges
    probability_add = []  # the probabilities of the pairs of nodes to be added
    score = 0.0  # the score of each pair of nodes in link prediction model
    total_score_without_edge = 0.0  # the sum of scores of pairs of nodes without edge

    #  calculate the score of each pair of nodes
    for i, elei in enumerate(list(network.nodes(), 1)):
        for j, elej in enumerate(list(network.nodes(), 1)):

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
    # select edges to be added according to probabilities
    edges_add = calculate_param.prob_select_distinct(nodes_pair_without_edge, probability_add, num_add)
    for a, b, c in edges_add:
        network.add_edge(a, b)  # add selected edges

    return True


def L_P_WAA(network):
    num_add = 0  # the number of egdes to be added
    nodes_pair = []  # the pairs of nodes with edges and without edges
    probability_add = []  # the probabilities of the pairs of nodes to be added
    score = 0.0  # the score of each pair of nodes in link prediction model
    total_score = 0.0  # the sum of scores of pairs of nodes with edge and without edge

    #  calculate the score of each pair of nodes
    for i, elei in enumerate(list(network.nodes()), 1):
        for j, elej in enumerate(list(network.nodes()), 1):

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

    for a, b, c in edges_add:
        network.add_edge(a, b)  # add selected edges

    return True


def L_P_RA(network):
    num_add = 0  # the number of egdes to be added
    nodes_pair_without_edge = []  # the pairs of nodes without edges
    probability_add = []  # the probabilities of the pairs of nodes to be added
    score = 0.0  # the score of each pair of nodes in link prediction model
    total_score_without_edge = 0.0  # the sum of scores of pairs of nodes without edge

    #  calculate the score of each pair of nodes
    for i, elei in enumerate(list(network.nodes(), 1)):
        for j, elej in enumerate(list(network.nodes(), 1)):

            if i >= j:
                continue
            if not network.has_edge(elei, elej):
                try:
                    pre = nx.resource_allocation_index(network, [(elei, elej)])
                    for u, v, s in pre:
                        score = s
                except:
                    continue
                total_score_without_edge += score
                nodes_pair_without_edge.append((elei, elej, score))

    for a, b, c in nodes_pair_without_edge:
        probability_add.append(c / total_score_without_edge)  # calculate the probabilities of edges to be added
    # select edges to be added according to probabilities
    edges_add = calculate_param.prob_select_distinct(nodes_pair_without_edge, probability_add, num_add)
    for a, b, c in edges_add:
        network.add_edge(a, b)  # add selected edges

    return True


def L_P_WRA(network):
    num_add = 0  # the number of egdes to be added
    nodes_pair = []  # the pairs of nodes with edges and without edges
    probability_add = []  # the probabilities of the pairs of nodes to be added
    score = 0.0  # the score of each pair of nodes in link prediction model
    total_score = 0.0  # the sum of scores of pairs of nodes with edge and without edge

    #  calculate the score of each pair of nodes
    for i, elei in enumerate(list(network.nodes()), 1):
        for j, elej in enumerate(list(network.nodes()), 1):

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
                    score += w_elei_z_elej / total_w_x_z
            except:
                continue
            total_score += score
            nodes_pair.append((elei, elej, score))

    for a, b, c in nodes_pair:
        probability_add.append(c / total_score)  # calculate the probabilities of edges to be added
    # select edges to be added according to probabilities
    edges_add = calculate_param.prob_select(nodes_pair, probability_add, num_add)

    for a, b, c in edges_add:
        network.add_edge(a, b)  # add selected edges

    return True


def L_P_KatzIndex(network, max_length):
    num_add = 0  # the number of egdes to be added
    nodes_pair_without_edge = []  # the pairs of nodes without edges
    probability_add = []  # the probabilities of the pairs of nodes to be added
    score = 0.0  # the score of each pair of nodes in link prediction model
    score_old = 0.0  # the last score of each pair of nodes in link prediction model
    total_score_without_edge = 0.0  # the sum of scores of pairs of nodes without edge
    B = 0.1  # a free parameter to control path weights. The longer the path is, the less contribution the path made to the similarity

    #  calculate the score of each pair of nodes
    for i, elei in enumerate(list(network.nodes())):
        for j, elej in enumerate(list(network.nodes())):

            if i >= j:
                continue
            if not network.has_edge(elei, elej):
                try:
                    for l in range(2, max_length):
                        score_old = score
                        score += pow(B, l) * pow(nx.to_numpy_array(network), l)[i][j]
                        if is_katz_converge(score, score_old):
                            break
                except:
                    continue
                total_score_without_edge += score
                nodes_pair_without_edge.append((elei, elej, score))

    for a, b, c in nodes_pair_without_edge:
        probability_add.append(c / total_score_without_edge)  # calculate the probabilities of edges to be added
    # select edges to be added according to probabilities
    edges_add = calculate_param.prob_select_distinct(nodes_pair_without_edge, probability_add, num_add)
    for a, b, c in edges_add:
        network.add_edge(a, b)  # add selected edges

    return True


def L_P_WKatzIndex(network):
    num_add = 0  # the number of egdes to be added
    nodes_pair = []  # the pairs of nodes with edges and without edges
    probability_add = []  # the probabilities of the pairs of nodes to be added
    score = 0.0  # the score of each pair of nodes in link prediction model
    score_old = 0.0  # the last score of each pair of nodes in link prediction model
    total_score = 0.0  # the sum of scores of pairs of nodes with edge and without edge
    B = 0.1  # a free parameter to control path weights. The longer the path is, the less contribution the path made to the similarity

    #  calculate the score of each pair of nodes
    for i, elei in enumerate(list(network.nodes())):
        for j, elej in enumerate(list(network.nodes())):

            if i >= j:
                continue
            try:
                for l in range(1, max_length):
                    score_old = score
                    A_matrix = nx.adjacency_matrix(network, None, 'weight').A
                    A_elei_elej_l = pow(A_matrix, l)[i][j]
                    score += pow(B, l) * A_elei_elej_l
                    if is_katz_converge(score, score_old):
                        break
            except:
                continue
            total_score += score
            nodes_pair.append((elei, elej, score))

    for a, b, c in nodes_pair:
        probability_add.append(c / total_score)  # calculate the probabilities of edges to be added
    # select edges to be added according to probabilities
    edges_add = calculate_param.prob_select(nodes_pair, probability_add, num_add)
    for a, b, c in edges_add:
        network.add_edge(a, b)  # add selected edges

    return True


def is_katz_converge(katz, katz_old, eps=1e-4):
    if abs(katz - katz_old) >= eps:
        return False
    return True


def L_P_SimRank(network, C=0.8, max_iter=100):
    # C: float, 0< C <=1, it is the decay factor which represents the relative importance between in-direct neighbors and direct neighbors.
    # max_iter: integer, the number specifies the maximum number of iterations for simrank
    num_add = 0  # the number of egdes to be added
    nodes_pair_without_edge = []  # the pairs of nodes without edges
    probability_add = []  # the probabilities of the pairs of nodes to be added
    score = 0.0  # the score of each pair of nodes in link prediction model
    total_score_without_edge = 0.0  # the sum of scores of pairs of nodes without edge

    # init.vars
    sim = defaultdict(list)  # the similarity between two nodes
    sim_old = defaultdict(list)  # the similarity between two nodes in last recursion
    for n in network.nodes():
        sim[n] = defaultdict(int)
        sim[n][n] = 1
        sim_old[n] = defaultdict(int)
        sim_old[n][n] = 0

    # recursively calculate simrank
    for iter_ctr in range(max_iter):
        if _is_sim_converge(sim, sim_old):
            break
        # calculate the score of each pair of nodes
        sim_old = copy.deepcopy(sim)
        for i, elei in enumerate(list(network.nodes()), 1):
            for j, elej in enumerate(list(network.nodes()), 1):
                if i == j:
                    continue
                try:
                    s_elei_elej = 0.0
                    for u in network.neighbors(elei):
                        for v in network.neighbors(elej):
                            s_elei_elej += sim_old[u][v]
                    sim[elei][elej] = (C * s_elei_elej / (len(network.neighbors(elei)) * len(network.neighbors(elej))))
                except:
                    continue

    for i, elei in enumerate(list(network.nodes()), 1):
        for j, elej in enumerate(list(network.nodes()), 1):
            if i >= j:
                continue
            if not network.has_edge(elei, elej):
                try:
                    score = sim[elei][elej]
                except:
                    continue
                total_score_without_edge += score
                nodes_pair_without_edge.append((elei, elej, score))

    for a, b, c in nodes_pair_without_edge:
        probability_add.append(c / total_score_without_edge)  # calculate the probabilities of edges to be added
    # select edges to be added according to probabilities
    edges_add = calculate_param.prob_select_distinct(nodes_pair_without_edge, probability_add, num_add)
    for a, b, c in edges_add:
        network.add_edge(a, b)  # add selected edges
    return True


def L_P_WSimRan(network, C=0.8, max_iter=100):
    # C: float, 0< C <=1, it is the decay factor which represents the relative importance between in-direct neighbors and direct neighbors.
    # max_iter: integer, the number specifies the maximum number of iterations for simrank
    num_add = 0  # the number of egdes to be added
    nodes_pair = []  # the pairs of nodes with edges and without edges
    probability_add = []  # the probabilities of the pairs of nodes to be added
    score = 0.0  # the score of each pair of nodes in link prediction model
    total_score = 0.0  # the sum of scores of pairs of nodes with edge and without edge

    # init.vars
    sim = defaultdict(list)  # the similarity between two nodes
    sim_old = defaultdict(list)  # the similarity between two nodes in last recursion
    for n in network.nodes():
        sim[n] = defaultdict(int)
        sim[n][n] = 1
        sim_old[n] = defaultdict(int)
        sim_old[n][n] = 0

    # recursively calculate simrank
    for iter_ctr in range(max_iter):
        if _is_sim_converge(sim, sim_old):
            break
        # calculate the score of each pair of nodes
        sim_old = copy.deepcopy(sim)
        for i, elei in enumerate(list(network.nodes()), 1):
            for j, elej in enumerate(list(network.nodes()), 1):
                if i == j:
                    continue
                try:
                    s_elei_elej = 0.0
                    total_w_z_elei = 0
                    total_w_z_elej = 0
                    for u in network.neighbors(elei):
                        for v in network.neighbors(elej):
                            s_elei_elej += sim_old[u][v]
                    for z in network.neighbors(elei):
                        w_z_elei = network.get_edge_data(z, elei).get('weight')
                        total_w_z_elei += w_z_elei
                    for z in network.neighbors(elej):
                        w_z_elej = network.get_edge_data(z, elej).get('weight')
                        total_w_z_elej += w_z_elej
                    sim[elei][elej] = (C * s_elei_elej / total_w_z_elei * total_w_z_elej)
                except:
                    continue

    for i, elei in enumerate(list(network.nodes()), 1):
        for j, elej in enumerate(list(network.nodes()), 1):
            if i >= j:
                continue
            try:
                score = sim[elei][elej]
            except:
                continue
            total_score += score
            nodes_pair.append((elei, elej, score))

    for a, b, c in nodes_pair:
        probability_add.append(c / total_score)  # calculate the probabilities of edges to be added
    # select edges to be added according to probabilities
    edges_add = calculate_param.prob_select(nodes_pair, probability_add, num_add)
    for a, b, c in edges_add:
        network.add_edge(a, b)  # add selected edges

    return True


def _is_sim_converge(s1, s2, eps=1e-4):
    for i in s1.keys():
        for j in s1[i].keys():
            if abs(s1[i][j] - s2[i][j]) >= eps:
                return False
    return True


def L_P_ACT(network):
    num_add = 0  # the number of egdes to be added
    nodes_pair_without_edge = []  # the pairs of nodes with edges and without edges
    probability_add = []  # the probabilities of the pairs of nodes to be added
    score = 0.0  # the score of each pair of nodes in link prediction model
    total_score_without_edge = 0.0  # the sum of scores of pairs of nodes without edge

    #  calculate the score of each pair of nodes
    for i, elei in enumerate(list(network.nodes(), 0)):
        for j, elej in enumerate(list(network.nodes(), 0)):

            if i >= j:
                continue
            if not network.has_edge(elei, elej):
                try:
                    L = nx.laplacian_matrix(network).A
                    L_aa = L[i][i]
                    L_bb = L[j][j]
                    L_ab = L[i][j]  # 修改L[a][b]变为L[i][j]
                    score = 1 / L_aa + L_bb - 2 * L_ab
                except:
                    continue
                total_score_without_edge += score
                nodes_pair_without_edge.append((elei, elej, score))

    for a, b, c in nodes_pair_without_edge:
        probability_add.append(c / total_score_without_edge)  # calculate the probabilities of edges to be added
    # select edges to be added according to probabilities
    edges_add = calculate_param.prob_select_distinct(nodes_pair_without_edge, probability_add, num_add)
    for a, b, c in edges_add:
        network.add_edge(a, b)  # add selected edges

    return True


def L_P_WACT(network):
    num_add = 0  # the number of egdes to be added
    nodes_pair = []  # the pairs of nodes with edges and without edges
    probability_add = []  # the probabilities of the pairs of nodes to be added
    score = 0.0  # the score of each pair of nodes in link prediction model
    total_score = 0.0  # the sum of scores of pairs of nodes with edge and without edge

    #  calculate the score of each pair of nodes
    for i, elei in enumerate(list(network.nodes(), 0)):
        for j, elej in enumerate(list(network.nodes(), 0)):
            if i >= j:
                continue
            try:
                L = nx.laplacian_matrix(network, None, 'weight').A
                L_aa = L[i][i]
                L_bb = L[j][j]
                L_ab = L[i][j]
                score = 1 / L_aa + L_bb - 2 * L_ab
            except:
                continue
            total_score += score
            nodes_pair.append((elei, elej, score))

    for a, b, c in nodes_pair:
        probability_add.append(c / total_score)  # calculate the probabilities of edges to be added
    # select edges to be added according to probabilities
    edges_add = calculate_param.prob_select(nodes_pair, probability_add, num_add)
    for a, b, c in edges_add:
        network.add_edge(a, b)  # add selected edges

    return True
