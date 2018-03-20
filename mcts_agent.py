import numpy as np


class Node():
    def __init__(self, state, parent):
        self.state = state
        self.children = {}
        self.parent = parent
        self.reward = 0
        self.visits = 0


class MCTSAgent():
    def __init__(self, action_space):
        self.action_space = action_space
    
    def act(self, state, env, search_size=30):
        original_state = env.unwrapped.clone_full_state()

        root = Node(state, parent=None)
        for _ in range(search_size):
            # Selection/Expansion: Find a new leaf node using the Tree Policy
            leaf = self.explore_to_leaf(root, env)

            # Simulation: Perform a "playout" or "rollout" with the Default Policy
            playout_reward = self.play_to_end(leaf, env)

            # Backup: Add to the reward in all parent nodes
            while leaf is not None:
                leaf.reward += playout_reward
                leaf.visits += 1
                leaf = leaf.parent

            # Put things back the way they were
            env.unwrapped.restore_full_state(original_state)

        # Select the best action from among the root's children
        best_action = 0
        best_score = -float('inf')
        for action, child in root.children.items():
            score = child.reward / child.visits
            if score > best_score:
                best_score = score
                best_action = action
        print('Best action: {} with expected reward {}'.format(best_action, best_score))
        return best_action
    
    def explore_to_leaf(self, root, env):
        # This is the part where we use the UCB formula
        node = root
        while True:
            action = self.tree_policy(node)
            state, reward, done, _ = env.step(action)
            if node.children.get(action) is None:
                # Expansion: Add this node to the tree
                node.children[action] = Node(state, parent=node)
                return node.children[action]
            node = node.children[action]

    def tree_policy(self, node):
        # First try any unexplored action
        for a in range(self.action_space.n):
            if a not in node.children:
                return a
        # Then try the UCB-optimal action
        def ucb(node):
            return node.reward / node.visits + np.sqrt(np.log(node.parent.visits) / node.visits)
        return max(node.children, key=lambda a: ucb(node.children[a]))

    def play_to_end(self, leaf, env, depth_limit=40):
        # This is what you call a 'simulation', a 'playout', or a 'rollout'
        cumulative_reward = 0
        depth = 0
        while True:
            action = self.action_space.sample()  # TODO: Default Policy
            state, reward, done, _ = env.step(action)
            cumulative_reward += reward
            # Heuristic: Stop search at any reward
            if reward != 0:
                done = True
            depth += 1
            if depth > depth_limit:
                done = True
            if done:
                break
        return cumulative_reward
