import time
import node

class MonteCarloTreeSearch(object):

    def __init__(self, game_state):
        """
        MonteCarloTreeSearchNode
        Parameters
        ----------
        game_state : monte-carlo-tree-search.tree.game_state
        """
        self.root = MonteCarloTreeSearchNode(game_state)

    def best_action(self, simulations_number=None, total_simulation_seconds=None):
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
                v = self._tree_policy()
                reward = v.rollout()
                v.backpropagate(reward)
                if time.time() > end_time:
                    break
        else :
            for _ in range(0, simulations_number):            
                v = self._tree_policy()
                reward = v.rollout()
                v.backpropagate(reward)
        # to select best child go for exploitation only
        return self.root.best_child(c_param=0.)

    def _tree_policy(self):
        """
        selects node to run rollout/playout for
        Returns
        -------
        """
        current_node = self.root
        while not current_node.is_terminal_node():
            # Current node will have all legal moves generated in initialization.
            if not len(current_node.untried_actions) == 0:
                # get next untried node from expansion
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node