import networkx as nx
import linecache
import calculate_param
import math
# import matplotlib.pyplot as plt


def DynamicNetwork(filename, start, end, network=None):
    # file_read = open(filename, 'r')
    if not network is None:
        nodes = set(network.nodes())
        edges = set(network.edges())
        G = network
    else:
        nodes = set()
        edges = set()
        G = nx.Graph(graphname=filename)

    print("start to build graph from {0} to {1}...".format(start, end))
    # build the graph from input file
    for i in range(start, end + 1):
        line_str = linecache.getline(filename, i)
        line = line_str.strip().strip('\n').split(' ')
        # print("{0}th line: {1}".format(i, line))
        node1 = int(line[0])
        node2 = int(line[1])
        timestamp = int(line[2])
        # add nodes to the graph
        G.add_nodes_from([node1, node2])
        nodes.update([node1, node2])
        # add edge to the graph, multiedge add to weight of the existing edge
        if G.has_edge(node1, node2):
            cur_weight = G[node1][node2]['weight']
            # G.edges[node1, node2]['weight'] = 1
            G[node1][node2]['weight'] = cur_weight + 1
        else:
            G.add_edge(node1, node2, weight=1)
        edges.add((node1, node2))
    print("graph building finished.")
    return G


def L_P_WCN(network, num_add):
    #num_add = 10  # the number of egdes to be added
    nodes_pair = []  # the pairs of nodes with edges and without edges
    probability_add = []  # the probabilities of the pairs of nodes to be added
    score = 0.0  # the score of each pair of nodes in link prediction model
    total_score = 0.0  # the sum of scores of pairs of nodes without edge and with edge

    #  calculate the score of each pair of nodes
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
    print("edges_add:")
    print(edges_add)

    '''
    for a, b, c in edges_add:
        network.add_edge(a, b)  # add selected edges
    '''
    return edges_add


def L_P_WAA(network, num_add):
    #num_add = 0  # the number of egdes to be added
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

    for a, b, c in edges_add:
        network.add_edge(a, b)  # add selected edges

    return edges_add


if __name__ == '__main__':
    dataset_file_name = "new_CollegeMsg.txt"
    start_timestamp = 1
    end_timestamp = 50000
    num_add = 1000
    # build Graph by networkx
    Gragh = DynamicNetwork(dataset_file_name, start_timestamp, end_timestamp)

    # nx.draw(Gragh, with_labels=True, font_weight='bold')
    print("Number of nodes:{0}".format(len(Gragh.nodes)))
    print(Gragh.edges.data())
    print(Gragh.nodes)

    # use model to predict edges to evaluate the recall
    print("using predict model to predict {0} edges...".format(num_add))
    # predict_list = L_P_WCN(Gragh, num_add)
    predict_list = L_P_WAA(Gragh, num_add)
    print("prediction finished.")

    print("get the {0} ground truth of edges following...".format(num_add))
    # get the ground truth of edges following
    ground_truth_edges_list = []
    for i in range(end_timestamp + 1, end_timestamp + num_add + 1):
        temp_line_str = linecache.getline(dataset_file_name, i)
        temp_line = temp_line_str.strip().strip('\n').split(' ')
        temp_node1 = int(temp_line[0])
        temp_node2 = int(temp_line[1])
        if Gragh.has_edge(temp_node1, temp_node2):
            weight = Gragh.edges[temp_node1, temp_node2]['weight']
        else:
            weight = 1
        ground_truth_edges_list.append((temp_node1, temp_node2, weight))

    print("Ground_truth_edge_list:")
    print(ground_truth_edges_list)

    print("Calculate the recall rate...")
    recall = calculate_param.calculate_recall(predict_list, ground_truth_edges_list)
    print("Recall rate is :{0}".format(recall))


