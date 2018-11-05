import networkx as nx
import calculate_param
import model_evaluator as me
import evolution_models_new as emn
import evolution_models_checked
import networks
import pandas as pd
import copy
import random

if __name__ == '__main__':
    # All arguments
    dataset_file_name_csv = r"../Datasets/CollegeMsg temporal dataset/new_CollegeMsg.csv"
    dataset_file_name_txt = r"../Datasets/CollegeMsg temporal dataset/new_CollegeMsg.txt"
    train_start_timestamp = 1
    train_end_timestamp = 3000  # 用时间戳来计算而不是数据集文件的行数(有可能一个时间戳增加了多条边)

    evaluate_interval = 100  # 预测的时间戳区间，从而决定会有多少条边，来评估模型的契合程度
    evaluate_edges_num = 0  # 预测多少条边来评估模型是否契合，应该是不确定的，取决于划分的时间戳之间新增了多少条边

    evaluate_timestamp = 3100  # 评估模型契合度的时刻
    take_action_timestamp = 3100  # 执行增边策略的时刻
    compare_gain_timestamp = 3500  # 比较中心度提升的时刻

    num_add = 100  # 应该是不确定的，取决于划分的时间戳之间新增了多少条边
    best_model_index = -1

    # read input to pandas dataframe
    dataset_df = pd.read_csv(filepath_or_buffer=dataset_file_name_csv)

    # build Graph by networkx
    # Graph = networks.build_dynamic_network_from_txt(dataset_file_name, train_start_timestamp, train_end_timestamp)
    Graph_original = networks.build_dynamic_network_from_dataframe(dataset_df, train_start_timestamp,
                                                                   train_end_timestamp)

    # get the ground truth of edges following
    ground_truth_edges_df = dataset_df[(dataset_df['timestamp'] > train_end_timestamp) & (
            dataset_df['timestamp'] <= train_end_timestamp + evaluate_interval)]
    evaluate_edges_num = len(ground_truth_edges_df)
    print("get the {0} ground truth of edges following...".format(evaluate_edges_num))

    ground_truth_edges_list = []
    for idx in ground_truth_edges_df.index:
        temp_node1 = ground_truth_edges_df.loc[idx, 'node1']
        temp_node2 = ground_truth_edges_df.loc[idx, 'node2']
        if Graph_original.has_edge(temp_node1, temp_node2):
            temp_weight = Graph_original.edges[temp_node1, temp_node2]['weight']
        else:
            temp_weight = 1
        ground_truth_edges_list.append((temp_node1, temp_node2, temp_weight))
    print("Ground_truth_edge_list:")
    print(ground_truth_edges_list)
    '''
    for i in range(train_end_timestamp + 1, train_end_timestamp + evaluate_edges_num + 1):
        temp_line_str = linecache.getline(dataset_file_name, i)
        temp_line = temp_line_str.strip().strip('\n').split(' ')
        temp_node1 = int(temp_line[0])
        temp_node2 = int(temp_line[1])
        if Gragh.has_edge(temp_node1, temp_node2):
            weight = Gragh.edges[temp_node1, temp_node2]['weight']
        else:
            weight = 1
        ground_truth_edges_list.append((temp_node1, temp_node2, weight))
    '''
    '''
    # use model to predict edges to evaluate the recall rate
    print("using predict model to predict {0} edges...".format(evaluate_edges_num))
    # predict_list = evolution_models_checked.L_P_WCN(Gragh, evaluate_edges_num)
    predict_list = evolution_models_checked.L_P_WAA(Graph_original, evaluate_edges_num)
    print("prediction finished.")

    # calculate the recall rate of this model
    print("Calculate the recall rate...")
    recall = calculate_param.calculate_recall(predict_list, ground_truth_edges_list)
    print("Recall rate is :{0}".format(recall))
    '''
    # Automatically evaluate every model
    best_model_index = me.find_best_fit_model(Graph_original, True, evaluate_edges_num, ground_truth_edges_list)
    print("best model is: {0}->{1}".format(best_model_index, emn.weighted_model_map[best_model_index]))

    # add those edges to the graph to synchronize graph to the evaluate timestamp
    Graph_original = networks.add_edges_to_graph_from_list(Graph_original, ground_truth_edges_list)
    # prepare a copy of current graph for taking action
    Graph_new = copy.deepcopy(Graph_original)

    # now, assume that we have a model fit the graph most, say AA
    # get the growth of the graph between evaluate_timestamp and compare_gain_timestamp
    normal_evolution_edges_df = dataset_df[(dataset_df['timestamp'] > evaluate_timestamp) & (
            dataset_df['timestamp'] <= compare_gain_timestamp)]
    normal_evolution_edges_list = []
    for idx in normal_evolution_edges_df.index:
        temp_node1 = normal_evolution_edges_df.loc[idx, 'node1']
        temp_node2 = normal_evolution_edges_df.loc[idx, 'node2']
        if Graph_original.has_edge(temp_node1, temp_node2):
            temp_weight = Graph_original.edges[temp_node1, temp_node2]['weight']
        else:
            temp_weight = 1
        normal_evolution_edges_list.append((temp_node1, temp_node2, temp_weight))
    normal_evolution_edges_num = len(normal_evolution_edges_list)

    # randomly choose a user/node
    node_list = list(Graph_new.nodes)
    chosen_node = node_list[random.randrange(0, len(node_list))]
    print("the chosen node is {0}".format(chosen_node))
    neighbors_of_chosen_node = nx.neighbors(Graph_new, chosen_node)
    print("the existing neighbors of chosen node:")
    print(list(neighbors_of_chosen_node))

    # record the centrality before take action
    centrality_before_action = nx.closeness_centrality(Graph_original, chosen_node)
    print("the centrality of chosen_node:{0} before take action is {1}".format(chosen_node, centrality_before_action))

    # original graph grows to compare_gain_timestamp, prepare for compare increase of chosen node
    Graph_original = networks.add_edges_to_graph_from_list(Graph_original, normal_evolution_edges_list)

    # record the centrality after graph's natural evolution
    centrality_after_natural_evolution = nx.closeness_centrality(Graph_original, chosen_node)
    print("the centrality of chosen_node:{0} after natural evolution is {1}".format(chosen_node,
                                                                                    centrality_after_natural_evolution))

    # take action of adding edge to chosen node
    # begin to try adding different edges to see the change of centrality
    edges_add_to_chosen_node = []
    best_centrality = centrality_before_action
    best_edge = None
    print("starting to try add different edge to chosen node:{0}".format(chosen_node))
    for node in Graph_new.nodes:
        if (node not in neighbors_of_chosen_node) and (node != chosen_node):
            print("\ntry add edge({0},{1}) to the graph_new.".format(chosen_node, node))
            # add an test edge to the graph
            Graph_new.add_edge(chosen_node, node, weight=1)
            # prepare to evolute the graph by model predicting, evolute same number of edges
            print("predicting and evoluting...")
            # predict_evolute_edges = evolution_models_checked.L_P_WAA(Graph_new, normal_evolution_edges_num)
            predict_evolute_edges = me.take_one_model_to_predict(Graph_new, True, normal_evolution_edges_num,
                                                                 best_model_index)
            # evolute
            Graph_new = networks.add_edges_to_graph_from_list(Graph_new, predict_evolute_edges)

            centrality_after_predict_evolution = nx.closeness_centrality(Graph_new, chosen_node)
            print("the centrality of chosen_node:{0} after natural evolution is {1}".format(chosen_node,
                                                                                            centrality_after_predict_evolution))
            if centrality_after_predict_evolution > best_centrality:
                best_centrality = centrality_after_predict_evolution
                best_edge = (chosen_node, node, 1)

            # remove changes of the graph
            Graph_new = networks.remove_edges_from_graph_from_list(Graph_new, predict_evolute_edges)
            Graph_new.remove_edge(chosen_node, node)

    print("\n\nfinal result:")
    print("the centrality of chosen_node:{0} before take action is {1}".format(chosen_node, centrality_before_action))
    print("the centrality of chosen_node:{0} after natural evolution is {1}".format(chosen_node,
                                                                                    centrality_after_natural_evolution))
    print("the best centrality of chosen_node:{0} after taking different action is {1}".format(chosen_node,
                                                                                               best_centrality))
    print("the best edge to add is ({0}, {1})".format(best_edge[0], best_edge[1]))
