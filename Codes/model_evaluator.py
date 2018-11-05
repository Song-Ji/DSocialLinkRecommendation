import evolution_models_new as emn
import networkx as nx
import calculate_param as cp
import math

prob_select_times = 100


def find_best_fit_model(graph: nx.Graph, weighted: bool, num_predict: int, ground_truth_list: list):
    if graph is None:
        raise Exception("graph is None!")
    if num_predict is None:
        raise Exception("num_predict is None! Please set the parameter!")

    best_model_index = 0
    best_recall_rate = 0.0
    # weighted graph
    if weighted:
        # use every model to predict and calculate recall rate to find the best one
        model_num = emn.weighted_model_num
        print("total number of models:{0}, start evaluating.".format(model_num))
        average_recall_rate_of_every_model = {}
        for index in emn.weighted_model_map.keys():
            # call different model dynamically
            print("try {0}th model {1}:".format(index, emn.weighted_model_map[index]))
            nodes_pair_list, probability_list = eval("emn." + emn.weighted_model_map[index])(graph)

            print("repeat prob_select {0} times to calculate average recall rate...".format(prob_select_times))
            recall_rate_list = []
            # repeat prob_select to calculate average recall rate for every model
            for i in range(0, prob_select_times):
                print("the {0}th time of selection:".format(i+1))
                predict_edge_list = cp.prob_select(nodes_pair_list, probability_list, num_predict)
                recall_rate = cp.calculate_recall(predict_edge_list, ground_truth_list)
                recall_rate_list.append(recall_rate)

            average_recall_rate = float(sum(recall_rate_list)) / len(recall_rate_list)
            print("average recall rate for {0} is {1}\n".format(emn.weighted_model_map[index], average_recall_rate))
            # record every model's recall rate
            average_recall_rate_of_every_model[index] = average_recall_rate

            if average_recall_rate > best_recall_rate:
                print("{0} > {1}, better model found:{2}:{3}\n".format(average_recall_rate, best_recall_rate, index,
                                                                     emn.weighted_model_map[index]))
                best_model_index = index
                best_recall_rate = average_recall_rate
    # unweighted TODO
    else:
        pass

    return best_model_index


def take_one_model_to_predict(graph: nx.Graph, weighted: bool, num_predict: int, model_index: int):
    if graph is None:
        raise Exception("graph is None!")
    if model_index is None or model_index == '':
        raise Exception("model_index is not valid!")

    if weighted:
        print("use {0} model to predict {1} edges...".format(emn.weighted_model_map[model_index], num_predict))
        nodes_pair_list, probability_list = eval("emn."+emn.weighted_model_map[model_index])(graph)
        edges_predict = cp.prob_select(nodes_pair_list, probability_list, num_predict)
    else:
        nodes_pair_list, probability_list = eval("emn."+emn.unweighted_model_map[model_index])(graph)
        edges_predict = cp.prob_select(nodes_pair_list, probability_list, num_predict)

    if edges_predict is None:
        raise Exception("model predicting error!")
    else:
        return edges_predict
