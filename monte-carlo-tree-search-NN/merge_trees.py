def merge_trees(tree1, tree2):
    """ Merges tree2 into tree1. """
    

    for key, value in tree2._number_of_visits_states_actions.items():
        if key in tree1._number_of_visits_states_actions:
            tree1._number_of_visits_states_actions[key] += value
        else:
            tree1._number_of_visits_states_actions[key] = value

    # Merge state-action q-values. Does a weighted average based on number of visits
    for key, value in tree2._quality_states_actions.items():
        if key in tree1._quality_states_actions:
            total_visits = tree1._number_of_visits_states_actions[key]
            tree1_contrib = tree1._quality_states_actions[key] * (total_visits - tree2._number_of_visits_states_actions[key])
            tree2_contrib = value * tree2._number_of_visits_states_actions[key]
            tree1._quality_states_actions[key] = (tree1_contrib + tree2_contrib) / total_visits
        else:
            tree1._quality_states_actions[key] = value

    # Merge state visits
    for key, value in tree2._number_of_visits_states.items():
        if key in tree1._number_of_visits_states:
            tree1._number_of_visits_states[key] += value
        else:
            tree1._number_of_visits_states[key] = value

    # Merge terminal states
    tree1._is_terminal_states.update(tree2._is_terminal_states)

    # Merge legal actions per state
    tree1._legal_actions_states.update(tree2._legal_actions_states)

    # Merge current player and previous player per state
    tree1._current_player.update(tree2._current_player)

    return tree1