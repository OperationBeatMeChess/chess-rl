import numpy as np

import torch


class MonteCarloTreeSearch:

    def __init__(self, game_state, nnet):
        """
        Parameters
        ----------
        game_state : gym.Env
        """
        self.game_state = game_state
        self.nnet = nnet
        self._quality_states_actions = {}
        self._number_of_visits_states_actions = {}
        self._number_of_visits_states = {}
        self._is_terminal_states = {}
        self._legal_actions_states = {}

        self._current_player = {}
        self._previous_player = {}
    
    def reset(self):
        self._quality_states_actions = {}
        self._number_of_visits_states_actions = {}
        self._number_of_visits_states = {}
        self._is_terminal_states = {}
        self._legal_actions_states = {}

        self._current_player = {}
        self._previous_player = {}

    def get_action_probabilities(self, state):
        legal_actions = self._legal_actions_states[state]
        action_probs = {
            action: self._number_of_visits_states_actions.get((action, state), 1e-8)
            for action in legal_actions
        }
        total_visits = sum(action_probs.values())
        action_probs = {
            action: (count / total_visits) for action, count in action_probs.items()
        }
        best_action = max(action_probs, key=action_probs.get)
        return action_probs, best_action
    
    def search(self, init_state, init_obs, simulations_number=10_000):
        for itr in range(simulations_number):
            self.game_state.set_string_representation(init_state)
            self.search_iteration(self.game_state, game_obs=init_obs)

        self.game_state.set_string_representation(init_state)
        return self.get_action_probabilities(init_state)

    def search_iteration(self, game_state, game_obs, c_param=1.4):
        state = game_state.get_string_representation()

        if state not in self._is_terminal_states:
            self._is_terminal_states[state] = game_state.game_result()

        if self._is_terminal_states[state] is not None:
            # terminal node
            winner = self._is_terminal_states[state]
            predicted_outcome = 1.
            return winner, predicted_outcome

        if state not in self._legal_actions_states:
            # Leaf node
            self._legal_actions_states[state] = game_state.action_space.legal_actions
            self._number_of_visits_states[state] = 1e-8
            self._current_player[state] = game_state.current_player
            self._previous_player[state] = game_state.previous_player

            player_at_leaf = self._current_player[state]

            # 1 current state wins, -1 previous state wins
            with torch.no_grad():
                predicted_outcome = self.nnet(game_obs[0])[1].item()

            return player_at_leaf, predicted_outcome

        best_action, best_ucb = self.best_action(state, c_param=c_param)

        # Traverse to next node in tree
        game_state.skip_next_human_render()
        observation, reward, terminated, truncated, info = game_state.step(best_action)
        player_at_leaf, predicted_outcome = self.search_iteration(
            game_state=game_state,
            game_obs=observation)

        # result is -1 if previous player won, and 1 if current player won.
        result = predicted_outcome if player_at_leaf == self._current_player[state] else - \
            predicted_outcome if player_at_leaf == self._previous_player[state] else 0

        # Backpropogate the result
        if (best_action, state) in self._quality_states_actions:
            q_old = self._quality_states_actions[(best_action, state)]
            self._quality_states_actions[(best_action, state)] = q_old + result
            self._number_of_visits_states_actions[(best_action, state)] += 1
        else:
            self._quality_states_actions[(best_action, state)] = result
            self._number_of_visits_states_actions[(best_action, state)] = 1

        self._number_of_visits_states[state] += 1
        return player_at_leaf, predicted_outcome

    def best_action(self, state, c_param=1.4):
        best_ucb = -np.inf
        best_action = None
        
        # find highest ucb action
        N = self._number_of_visits_states[state]
        LOGN = np.log(N)
        for action in self._legal_actions_states[state]:

            if (action, state) in self._quality_states_actions:
                q = self._quality_states_actions[(action, state)]
                n = self._number_of_visits_states_actions[(action, state)]
                ucb = (q / n) + c_param * np.sqrt(LOGN / n)
            else:
                ucb = np.inf

            if ucb > best_ucb:
                best_ucb = ucb
                best_action = action

        return best_action, best_ucb
