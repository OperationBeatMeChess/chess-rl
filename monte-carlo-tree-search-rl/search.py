import numpy as np
from collections import defaultdict
from abc import ABC, abstractmethod
import gym

class MonteCarloTreeSearch(object):

    def __init__(self, game_state):
        """
        Parameters
        ----------
        game_state : gym.Env
        """
        self.game_state = game_state

        self._quality_states_actions = {}
        self._number_of_visits_states_actions = {}
        self._number_of_visits_states = {}
        self._is_terminal_states = {}
        self._legal_actions_states = {}

        self._current_player = {}
        self._previous_player = {}
    
    def search(self, init_state, simulations_number=1000):
        for itr in range(simulations_number):
            self.game_state.set_string_representation(init_state)
            self.search_iteration(self.game_state)

        self.game_state.set_string_representation(init_state)
        return self.best_action(init_state, c_param=0.)

    def search_iteration(self, game_state, c_param=1.4):
        
        state = game_state.get_string_representation()

        if state not in self._is_terminal_states:
            self._is_terminal_states[state] = game_state.game_result()
            
        if self._is_terminal_states[state] is not None:
            # terminal node
            return self._is_terminal_states[state]
        
        if state not in self._legal_actions_states:
            # Leaf node
            self._legal_actions_states[state] = game_state.action_space.legal_actions
            self._number_of_visits_states[state] = 1e-8
            self._current_player[state] = game_state.current_player
            self._previous_player[state] = game_state.previous_player
            # perform rollout
            done = False
            while not done:
                action = game_state.action_space.sample()
                observation, rew, done, info = game_state.step(action)
            return game_state.game_result()

        best_action = self.best_action(state, c_param=c_param)

        # Traverse to next node in tree
        game_state.step(best_action)
        r = self.search_iteration(game_state=game_state)
        # result is -1 if previous player won, and 1 if current player won.
        result = 1. if r==self._current_player[state] else -1. if r==self._previous_player[state] else 0

        # Backpropogate the result
        if (best_action, state) in self._quality_states_actions:
            q_old = self._quality_states_actions[(best_action, state)]
            self._quality_states_actions[(best_action, state)] = q_old + result
            self._number_of_visits_states_actions[(best_action, state)] += 1
        else:
            self._quality_states_actions[(best_action, state)] = result
            self._number_of_visits_states_actions[(best_action, state)] = 1

        self._number_of_visits_states[state] += 1
        return r

    def best_action(self, state, c_param=1.4):
        best_ucb = -np.inf
        best_action = None
        # find highest ucb action
        for action in self._legal_actions_states[state]:
            N = self._number_of_visits_states[state]

            if (action, state) in self._quality_states_actions:
                q = self._quality_states_actions[(action, state)]
                n = self._number_of_visits_states_actions[(action, state)]
                ucb = (q / n) + c_param * np.sqrt((np.log(N) / n))
            else:
                ucb = np.inf

            if ucb > best_ucb:
                best_ucb = ucb
                best_action = action

        return best_action
