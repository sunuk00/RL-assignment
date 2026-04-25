import numpy as np

def argmax(arr):
  return np.random.choice([idx for idx in range(len(arr)) if arr[idx] == arr.max()])

class FrozenLake:
  def __init__(self):
    self.width = 4
    self.height = 4
    self.states = [(x, y) for x in range(self.height) for y in range(self.width)]
    self.terminal_states = [(0, 2), (1, 0), (2, 1), (2, 3), (3, 3)]
    self.actions = [0, 1, 2, 3]
    self.actions_str = ['Up', 'Down', 'Left', 'Right']
    self.gamma = 1.0
    self.hole_states = [(0, 2), (1, 0), (2, 1), (2, 3)]
    self.goal_states = [(3, 3)]

  def transition_prob(self, state, action):

    if state in self.hole_states:
      return [(1.0, self.state, -2)]  #p(s'r|s,a), next state, reward

    elif state in self.goal_states:
      return [(1.0, self.state, 2)]  #p(s'r|s,a), next state, reward

    else:
      x = state[0]
      y = state[1]

      if action == 0:     # Up
        x = max(x-1, 0)

      elif action == 1:   # Down
        x = min(x+1, self.height-1)

      elif action == 2:   # Left
        y = max(y-1, 0)

      elif action == 3:   # Right
        y = min(y+1, self.width-1)


    next_state = (x, y)


    if next_state in self.hole_states:
      reward = -2

    elif next_state in self.goal_states:
      reward = +2

    else:
      reward = 0


    return [(1.0, next_state, reward)] #p(s'r|s,a), next state, reward


# Define the equiprobable random policy
def random_policy(env):
  '''
  Define equiprobable random policy
  '''
  policy = {state: np.zeros((len(env.actions))) for state in env.states}

  for state in env.states:
    if state in env.terminal_states:
      continue

    else:
      for action in env.actions:
        policy[state][action] = 1/len(env.actions)

  return policy


# Policy evaluation function
def policy_evaluation(env, policy, theta=1e-4):
  '''
  Policy evaluation code
  '''
  V = {state: 0.0 for state in env.states} # Initialize value function V(s) to 0 for all states

  iter = 0

  while True:

    delta = 0.0

    for state in env.states:
      if state in env.terminal_states:
        continue

      else:
        v_old = V[state]
        v_new = 0.0

        for action_idx, action_prob in enumerate(policy[state]):
          for trans_prob, next_state, reward in env.transition_prob(state, action_idx):
            v_new += action_prob * trans_prob * (reward + env.gamma * V[next_state]) #Bellman update rule


        delta = max(delta, abs(v_new - v_old)) #compute delta (update error)
        V[state] = float(v_new)

    iter += 1

    # Check for convergence
    if delta < theta:
      print('==========================================================')
      print(f'Total Iteration: {iter}')
      for state in env.states:
        V[state] = round(V[state], 2)
      break

  return V


# Procedure to generate episodes and print results
def main():
    env = FrozenLake()
    random_pol = random_policy(env) # Generate a random policy
    Value_grid = policy_evaluation(env, random_pol)

    # To display the value grid in a readable 4x4 format
    grid_values_display = np.zeros((env.height, env.width))
    for state, value in Value_grid.items():
        grid_values_display[state[0], state[1]] = value

    print("Value Function (V_pi) for Random Policy:")
    print(grid_values_display)


if __name__ == "__main__":
    main()