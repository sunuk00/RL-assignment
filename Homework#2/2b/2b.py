import numpy as np

def argmax(arr):
  return np.random.choice([idx for idx in range(len(arr)) if arr[idx] == arr.max()])

class ClimbingRobot:
  def __init__(self):
    self.states = ['Low', 'Middle', 'High']
    self.actions = [0, 1]
    self.actions_str = ['Run', 'Do not run']
    self.gamma = 0.8

  def transition_prob(self, state, action):
    #Run
    if action == 0:
      if state == 'Low':
        return [(0.5, 'Low', -2), (0.5, 'Middle', -2)] #p(s'r|s,a), next state, reward
      
      elif state == 'Middle':
        return [(0.5, 'Low', +2), (0.5, 'High', +2)] #p(s'r|s,a), next state, reward
      
      elif state == 'High':
        return [(0.5, 'Middle', +2), (0.5, 'High', +2)] #p(s'r|s,a), next state, reward

    #Do not run
    if action == 1:
      if state == 'Low':
        return [(1.0, 'Low', +1)] #p(s'r|s,a), next state, reward

      elif state == 'Middle':
        return [(1.0, 'Low', +4)] #p(s'r|s,a), next state, reward

      elif state == 'High':
        return [(1.0, 'Middle', +4)] #p(s'r|s,a), next state, reward


def value_iteration(env):
  '''
  Create the value iteration algorithm
  '''
  V = {state: 0.0 for state in env.states}
  theta = 1e-4
  
  # Initialize the optimal policy with the first action for all states
  optimal_policy = {state: env.actions[0] for state in env.states} 
  iter = 0

  while True:
    delta = 0.0

    for state in env.states:
      v_old = V[state]

      # q(s,a) calculation for all actions in the current state  
      action_values = []
      for action in env.actions:
        q_sa = 0.0
        for trans_prob, next_state, reward in env.transition_prob(state, action):
          q_sa += trans_prob * (reward + env.gamma * V[next_state])
        action_values.append(q_sa)

      # Find the best action and update the value function
      best_action_idx = int(np.argmax(action_values))
      V[state] = action_values[best_action_idx]
      
      # Update the optimal policy for the current state
      optimal_policy[state] = env.actions[best_action_idx]

      delta = max(delta, abs(v_old - V[state]))

    iter += 1

    if delta < theta:
      break

  return V, optimal_policy, iter


def main():
  env = ClimbingRobot()
  V, optimal_policy, iter = value_iteration(env)

  print('==========================================================')
  print(f'Total Iteration: {iter}')
  print('Optimal Value Function (V*):')
  for state in env.states:
    print(f'  V({state}) = {V[state]:.4f}')

  print('Optimal Policy (pi*):')
  for state in env.states:
    action = optimal_policy[state]
    print(f'  pi*({state}) = {env.actions_str[action]}')


if __name__ == '__main__':
  main()