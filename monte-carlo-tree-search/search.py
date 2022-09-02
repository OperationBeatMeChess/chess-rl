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
        self.root = MonteCarloTreeSearchNode(game_env)
        self.game_state = game_env
        self.init_state = self.game_state.get_string_representation()

    def step(self):
        """
        Takes a step using search and sets the root to be the best child found with search.
        This will maintain the search history for each node.
        """
        pass

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
                self.game_state.set_string_representation(self.init_state)
                v = self._select()
                winning_player, quality = v.simulation()
                v.backpropagate(winning_player, quality)
                if time.time() > end_time:
                    break
        else :
            for _ in range(0, simulations_number):            
                self.game_state.set_string_representation(self.init_state)
                v = self._select()
                winning_player, quality = v.simulation()
                v.backpropagate(winning_player, quality)
        # to select best child go for exploitation only
        child_node = self.root.select_child(c_param=0.)
        action = child_node.previous_action

        self.game_state.set_string_representation(self.init_state)
        return action

    def _select(self):
        """
        selects next node to run simulation/rollout/playout from. 
        -------
        """
        current_node = self.root
        while not current_node.is_leaf():
            current_node = current_node.select_child()
        current_node.expand()
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
        self.hash_state = self.game_state.get_string_representation()
        self.parent = parent
        self.children = []
        self.is_done = self.game_state.game_result() is not None
        
        self._current_player = self.game_state.current_player
        self._number_of_visits = 0.
        self._quality = defaultdict(int)
        self.previous_action = None

    @property
    def previous_player(self):      
        return -self.current_player

    @property
    def current_player(self):      
        return self._current_player

    @property
    def q(self):
        # Want quality value that makes the parent win because quality is found for each child
        wins = self._quality[self.parent.current_player]
        loses = self._quality[self.parent.previous_player]
        return wins - loses

    @property
    def n(self):
        return self._number_of_visits

    def ucb(self, c_param):
        if self.n == 0:
            return np.inf
        return (self.q / self.n) + c_param * np.sqrt((np.log(self.parent.n) / self.n))

    def simulation_policy(self, possible_moves):        
        # IDEA: Rollout done with lstm or transformer. 
        # forward pass if legal take most likely game out come otherwise rollout random moves.
        # complete this for every move of a game then store game sequence with reward, state, action tuples to retrain
        return possible_moves[np.random.randint(len(possible_moves))]

    # Rollout can use environment step. 
    def simulation(self):
        done = self.is_done
        quality = done
        while not done:
            possible_moves = self.game_state.action_space.legal_actions
            action = self.simulation_policy(possible_moves)
            observation, rew, done, info = self.game_state.step(action)
            quality += rew
        result = self.game_state.game_result()
        return result, quality

    def backpropagate(self, winning_player, quality):
        self._number_of_visits += 1.
        self._quality[winning_player] += abs(quality)
        # check it has a parent. Not a root.
        if self.parent: 
            self.parent.backpropagate(winning_player, quality)

    def expand(self):
        # at terminal node or already has already been expanded with children  
        if self.is_done or self.children:
            return

        self.game_state.set_string_representation(self.hash_state)

        # adds a child for every legal move with an initial 
        # quality value of infinity.
        for action in self.game_state.action_space.legal_actions:
            _ = self.game_state.step(action)
            child = MonteCarloTreeSearchNode(
                self.game_state, parent=self
            )
            child.previous_action = action
            self.children.append(child)
            self.game_state.set_string_representation(self.hash_state)
        

    def select_child(self, c_param=1.4):
        # at leaf return self (at terminal node or has no children)
        if self.is_leaf():
            return self
        children_weights = [c.ucb(c_param) for c in self.children]
        # if c_param==0:
        #     print('children:', list(zip([c.previous_action for c in self.children], children_weights)), '\n')
        #     print('children weights:', children_weights)
        child_node = self.children[np.argmax(children_weights)]
        return child_node 
    
    def is_leaf(self):
        return self.is_done or not self.children