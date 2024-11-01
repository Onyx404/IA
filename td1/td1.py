import numpy as np

class GridWorld:
    def __init__(self, n, gamma):
        self.n = n
        self.gamma = gamma
        self.V = np.zeros((n, n))  
        self.policy = np.zeros((n, n), dtype=int)  
        self.terminal_states = [(0, 0), (n - 1, n - 1)]  

    def step(self, state, action):
        i, j = state
        if state in self.terminal_states:
            return state, 0  

        if action == 0:  #U P
            next_state = (max(i - 1, 0), j)
        elif action == 1:  # RIGHT
            next_state = (i, min(j + 1, self.n - 1))
        elif action == 2:  # DOWN
            next_state = (min(i + 1, self.n - 1), j)
        elif action == 3:  # LEFT
            next_state = (i, max(j - 1, 0))
        
        reward = -1
        return next_state, reward

    
    def value_iteration(self, theta=1e-4):
        while True:
            delta = 0
            for i in range(self.n):
                for j in range(self.n):
                    if (i, j) in self.terminal_states:
                        continue

                    v = self.V[i, j]
                    values = []
                    for action in range(4):
                        (next_i, next_j), reward = self.step((i, j), action)
                        values.append(reward + self.gamma * self.V[next_i, next_j])
                    
                    self.V[i, j] = max(values)
                    delta = max(delta, abs(v - self.V[i, j]))

            if delta < theta:
                break

        for i in range(self.n):
            for j in range(self.n):
                if (i, j) in self.terminal_states:
                    continue
                values = [self.step((i, j), action)[1] + self.gamma * self.V[self.step((i, j), action)[0][0], self.step((i, j), action)[0][1]] for action in range(4)]
                self.policy[i, j] = np.argmax(values)

        return self.policy, self.V

    def policy_iteration(self, theta=1e-4):
        while True:
            while True:
                delta = 0
                for i in range(self.n):
                    for j in range(self.n):
                        if (i, j) in self.terminal_states:
                            continue
                        
                        v = self.V[i, j]
                        action = self.policy[i, j]
                        (next_i, next_j), reward = self.step((i, j), action)
                        self.V[i, j] = reward + self.gamma * self.V[next_i, next_j]
                        delta = max(delta, abs(v - self.V[i, j]))

                if delta < theta:
                    break

            policy_stable = True
            for i in range(self.n):
                for j in range(self.n):
                    if (i, j) in self.terminal_states:
                        continue
                    
                    old_action = self.policy[i, j]
                    values = [self.step((i, j), action)[1] + self.gamma * self.V[self.step((i, j), action)[0][0], self.step((i, j), action)[0][1]] for action in range(4)]
                    self.policy[i, j] = np.argmax(values)

                    if old_action != self.policy[i, j]:
                        policy_stable = False

            if policy_stable:
                break

        return self.policy, self.V
    
grid_world = GridWorld(n=4, gamma=0.9)

optimal_policy_value, optimal_value_function = grid_world.value_iteration()

optimal_policy_policy, optimal_value_function_policy = grid_world.policy_iteration()

print("Politique optimale (Value Iteration):\n", optimal_policy_value)
print("Fonction de valeur optimale (Value Iteration):\n", optimal_value_function)

print("Politique optimale (Policy Iteration):\n", optimal_policy_policy)
print("Fonction de valeur optimale (Policy Iteration):\n", optimal_value_function_policy)

