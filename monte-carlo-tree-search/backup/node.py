import numpy as np
from collections import defaultdict
from abc import ABC, abstractmethod


class MonteCarloTreeSearchNode(ABC):

    def __init__(self, game_state, parent=None):
        """
        Parameters
        ----------
        game_state : monte-carlo-tree-search.tree.game_state
        parent : MonteCarloTreeSearchNode
        """

        self.game_state = game_state
        self.parent = parent
        self.children = []
        
        self._number_of_visits = 0.
        self._results = defaultdict(int)
        self._untried_actions = None

        self.player1 = 1
        self.player2 = -1
        self._previous_player = None

    @property
    def previous_player(self):      
        if self._previous_player is None:
            # If root then player 1 starts
            if self.parent is None:
                self._previous_player = self.player1
            # If not root then the player of the parent is now previous player
            else:
                self._previous_player = self.parent.current_player
        return self._previous_player

    @property
    def current_player(self):      
        return -self.previous_player

    @property
    def q(self):
        wins = self._results[self.parent.game_state.current_player]
        loses = self._results[self.parent.game_state.previous_player]
        return wins - loses

    @property
    def n(self):
        return self._number_of_visits

    @property
    def untried_actions(self):      
        if self._untried_actions is None:
            self._untried_actions = self.game_state.get_legal_moves()
        return self._untried_actions

    def expand(self):
        action = self.untried_actions.pop()
        next_game_state = self.game_state.move(action)
        child_node = MonteCarloTreeSearchNode(
            next_game_state, parent=self
        )
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self): 
        return self.game_state.is_game_over()

    def rollout_policy(self, possible_moves):        
        return possible_moves[np.random.randint(len(possible_moves))]

    # Rollout can use environment step. 
    def rollout(self):
        current_rollout_game_state = self.game_state
        while not current_rollout_game_state.is_game_over():
            possible_moves = current_rollout_game_state.get_legal_moves()
            action = self.rollout_policy(possible_moves)
            current_rollout_game_state = current_rollout_game_state.move(action)
        return current_rollout_game_state.result

        # IDEA: Rollout done with lstm or transformer. 
        # forward pass if legal take most likely game out come otherwise rollout random moves.
        # complete this for every move of a game then store game sequence with reward, state, action tuples to retrain

    def backpropagate(self, result):
        self._number_of_visits += 1.
        # result = 1 or -1 for player1 or player2 e.g.
        # _results[1] = player1 win, _results[0] = draws, _results[-1] = player2 win
        self._results[result] += 1.
        # check it has a parent. Not a root.
        if self.parent: 
            self.parent.backpropagate(result)

    def best_child(self, c_param=1.4):
        choices_weights = [
            (c.q / c.n) + c_param * np.sqrt((2 * np.log(self.n) / c.n))
            for c in self.children
        ]
        return self.children[np.argmax(choices_weights)]

