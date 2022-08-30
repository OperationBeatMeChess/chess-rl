import numpy as np
from collections import defaultdict
from abc import ABC, abstractmethod
import gym
import copy 
import time

class MonteCarloTreeSearch(object):

    def __init__(self, game_env):
        """
        Parameters
        ----------
        game_env : gym.Env
        """
        # self.root = MonteCarloTreeSearchNode(copy.deepcopy(game_env))
        self.root = MonteCarloTreeSearchNode(game_env)

    def search(self, simulations_number=None, total_simulation_seconds=None):
        """
        Parameters
        ----------
        simulations_number : int
            number of simulations performed to get the best action
        total_simulation_seconds : float
            Amount of time the algorithm has to run. Specified in seconds
        Returns
        -------
        """

        if simulations_number is None :
            assert(total_simulation_seconds is not None)
            end_time = time.time() + total_simulation_seconds
            while True:
                v = self._select()
                winning_player, quality = v.simulation()
                v.backpropagate(winning_player, quality)
                if time.time() > end_time:
                    break
        else :
            for _ in range(0, simulations_number):            
                v = self._select()
                winning_player, quality = v.simulation()
                v.backpropagate(winning_player, quality)
        # to select best child go for exploitation only
        child_node = self.root.best_child(c_param=0.)
        action = child_node.previous_action
        return action

    def _select(self):
        """
        selects next node to run simulation/rollout/playout from. 
        -------
        """
        current_node = self.root
        at_leaf = False
        while not at_leaf:
            current_node, at_leaf = current_node.select_child()
        return current_node

class MonteCarloTreeSearchNode(ABC):

    def __init__(self, game_state, parent=None):
        """
        Parameters
        ----------
        game_state : gym.Env
        parent : MonteCarloTreeSearchNode
        """

        self.game_state = game_state
        self.parent = parent
        self.children = []
        
        self._number_of_visits = 0.
        self._quality = defaultdict(int)
        self._untried_children = None
        self.previous_action = None

    @property
    def previous_player(self):      
        return self.game_state.action_space.previous_player

    @property
    def current_player(self):      
        return self.game_state.action_space.current_player

    @property
    def q(self):
        wins = self._quality[self.parent.current_player]
        loses = self._quality[self.parent.previous_player]
        return wins - loses

    @property
    def n(self):
        return self._number_of_visits

    @property
    def untried_children(self):  
        # Current node will have all legal moves generated in initialization.    
        if self._untried_children is None:
            self._untried_children = self.game_state.action_space.legal_actions
        return self._untried_children

    def simulation_policy(self, possible_moves):        
        # IDEA: Rollout done with lstm or transformer. 
        # forward pass if legal take most likely game out come otherwise rollout random moves.
        # complete this for every move of a game then store game sequence with reward, state, action tuples to retrain
        return possible_moves[np.random.randint(len(possible_moves))]

    # Rollout can use environment step. 
    def simulation(self):
        current_simulation_game_state = copy.deepcopy(self.game_state)
        done, quality = current_simulation_game_state.is_done(), 0
        while not done:
            possible_moves = current_simulation_game_state.action_space.legal_actions
            action = self.simulation_policy(possible_moves)
            observation, rew, done, info = current_simulation_game_state.step(action)
            quality += rew
        return current_simulation_game_state.winning_player(), float(quality)

    def backpropagate(self, winning_player, quality):
        self._number_of_visits += 1.
        self._quality[winning_player] += abs(quality)
        # check it has a parent. Not a root.
        if self.parent: 
            self.parent.backpropagate(winning_player, quality)

    def select_child(self, c_param=1.4):
        is_leaf = None
        child_node = None

        if len(self.untried_children) != 0:
            is_leaf = True
            child_node, action = self._expand()
            return child_node, is_leaf

        child_node = self.best_child(c_param)
        is_leaf = child_node.game_state.is_done()
        return child_node, is_leaf 

    def best_child(self, c_param=1.4):
        children_weights = [
            (c.q / c.n) + c_param * np.sqrt((2 * np.log(self.n) / c.n))
            for c in self.children
        ]
        child_node = self.children[np.argmax(children_weights)]
        return child_node 

    def _expand(self):
        action = self.untried_children.pop()
        next_game_state = copy.deepcopy(self.game_state)
        observation, reward, terminal, info = next_game_state.step(action)

        child_node = MonteCarloTreeSearchNode(
            next_game_state, parent=self
        )
        child_node.previous_action = action
        self.children.append(child_node)
        return child_node, action