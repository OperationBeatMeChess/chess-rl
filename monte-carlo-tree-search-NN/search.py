import numpy as np
from collections import defaultdict
from abc import ABC, abstractmethod
import gym
from collections import deque

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

    def search(self, init_state, init_obs, simulations_number=10_000):
        for itr in range(simulations_number):
            self.game_state.set_string_representation(init_state)
            # length = 20
            # traversal_queue = deque(np.array([np.zeros(64) for i in range(length)]), maxlen=length)
            traversal_queue = None
            self.search_iteration(self.game_state, game_obs=init_obs, traversal_queue=traversal_queue)

        self.game_state.set_string_representation(init_state)
        return self.best_action(init_state, c_param=0.)

    def search_iteration(self, game_state, game_obs, traversal_queue=None, c_param=1.4):
        state = game_state.get_string_representation()

        if traversal_queue is not None:
            traversal_queue.append(game_obs[0].flatten())

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

            # player_at_leaf = game_obs[1] 
            player_at_leaf = self._current_player[state]

            # 1 current state wins, -1 previous state wins
            # predicted_outcome = np.random.rand()
            # _, predicted_outcome = model(game_obs[0])
            with torch.no_grad():
                _, predicted_outcome = self.nnet(game_obs[0])
            # predicted_outcome = self.nnet.predict(np.array(traversal_queue, dtype=np.uint8).reshape((20, 1, 64)))
            # predicted_outcome = self.nnet.predict(np.array(traversal_queue, dtype=np.uint8)[np.newaxis, ...])[-1]

            # print('current player:', player_at_leaf, 'player at start:', game_state.starting_player, 'predicted outcome:', predicted_outcome)
            # predicted_outcome = predicted_outcome  * 2 - 1
            return player_at_leaf, predicted_outcome

        best_action, best_ucb = self.best_action(state, c_param=c_param)

        # Traverse to next node in tree
        game_state.skip_next_human_render()
        observation, reward, terminated, truncated, info = game_state.step(best_action)
        player_at_leaf, predicted_outcome = self.search_iteration(
            game_state=game_state,
            game_obs=observation,
            traversal_queue=traversal_queue)

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
