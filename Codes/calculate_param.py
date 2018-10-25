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


def calculate_recall(predict_list, ground_truth_list):
    if len(predict_list) != len(ground_truth_list):
        print("exception:length of predict_list does not match ground_truth_list!")
        return
    else:
        total_edge_num = len(predict_list)
        hit_num = 0
        for i in range(0, total_edge_num):
            for j in range(0, total_edge_num):
                # correct ignore direction
                if (predict_list[i][0] == ground_truth_list[j][0] and predict_list[i][1] == ground_truth_list[j][1]) or (predict_list[i][0] == ground_truth_list[j][1] and predict_list[i][1] == ground_truth_list[j][0]):
                    hit_num += 1
                    print("correct edges:{0}--{1}".format(predict_list[i], ground_truth_list[j]))

        recall = float(hit_num)/float(total_edge_num)
        return recall
