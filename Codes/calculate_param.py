def prob_select(target_list, probability_list, num):

    selected_list = []
    i = 0
    while i < num:
        x = random.uniform(0,1)
        cumulative_probability = 0.0
        for item, item_probability in zip(target_list, probability_list):
            cumulative_probability += item_probability
            if x < cumulative_probability:
                selected_list.append(item)
                break
        i += 1
    #selected_list = list(set(selected))  # remove repeated items
    return selected_list



def prob_select_distinct(target_list, probability_list, num):

    selected_list = []
    while len(selected_list) < num:
        x = random.uniform(0,1)
        cumulative_probability = 0.0
        for item, item_probability in zip(target_list, probability_list):
            cumulative_probability += item_probability
            if x < cumulative_probability:
                selected_list.append(item)
                break
        selected_list = list(set(selected_list))  # remove repeated items

    return selected_list
