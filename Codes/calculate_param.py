import random


def prob_select(target_list, probability_list, num):
    selected_list = []
    i = 0
    while i < num:
        x = random.uniform(0, 1)
        cumulative_probability = 0.0
        for item, item_probability in zip(target_list, probability_list):
            cumulative_probability += item_probability
            if x < cumulative_probability:
                selected_list.append(item)
                break
        i += 1
    # selected_list = list(set(selected))  # remove repeated items
    return selected_list


def prob_select_distinct(target_list, probability_list, num):
    selected_list = []
    while len(selected_list) < num:
        x = random.uniform(0, 1)
        cumulative_probability = 0.0
        for item, item_probability in zip(target_list, probability_list):
            cumulative_probability += item_probability
            if x < cumulative_probability:
                selected_list.append(item)
                break
        selected_list = list(set(selected_list))  # remove repeated items

    return selected_list


def calculate_recall(predict_list: list, ground_truth_list: list):
    if len(predict_list) != len(ground_truth_list):
        raise Exception("length of predict_list does not match ground_truth_list!")
    else:
        total_edge_num = len(predict_list)
        hit_num = 0
        for truth_edge in ground_truth_list:
            for predict_edge in predict_list:
                # correct ignore direction
                if (predict_edge[0] == truth_edge[0] and predict_edge[1] == truth_edge[1]) or (predict_edge[0] == truth_edge[1] and predict_edge[1] == truth_edge[0]):
                    hit_num += 1
                    print("correct edges:{0}--{1}".format(truth_edge, predict_edge))
                    predict_list.remove(predict_edge)
                    break

        recall = float(hit_num) / float(total_edge_num)
        print("recall rate is:{0}\n".format(recall))
        return recall
