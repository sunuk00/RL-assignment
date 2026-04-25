import numpy as np

class Cardgame:
  def __init__(self, alpha):
    '''
    Intialize the class: define state space, action space, initital state, alpha, etc. (5 pts)'''
    self.state_space = [0, 1, 2, 3, 4, 5, "T_STOP", "T_BUST"]   #define state space
    self.action_space = ['draw', 'stop']                        #define action space

    self.start_state = 0      #Start state
    self.state = 0            #Current state
    self.non_terminal_states = [0, 1, 2, 3, 4, 5]

    self.alpha = alpha        #alpha => +2 / 1-alpha => +1

  def stop_reward(self, state):
    if state <= 3:
      return 0
    if state == 4:
      return 1
    return 2

  def transition_model(self, state, action):
    """
    Return explicit transition dynamics as a list of:
    (probability, next_state, reward, done)
    """
    if action == 'stop':
      return [(1.0, 'T_STOP', self.stop_reward(state), True)]

    if action == 'draw':
      transitions = []

      # card = 1 with probability (1-alpha)
      next_state_1 = state + 1
      if next_state_1 > 5:
        transitions.append((1 - self.alpha, 'T_BUST', -1, True))
      else:
        transitions.append((1 - self.alpha, next_state_1, 0, False))

      # card = 2 with probability alpha
      next_state_2 = state + 2
      if next_state_2 > 5:
        transitions.append((self.alpha, 'T_BUST', -1, True))
      else:
        transitions.append((self.alpha, next_state_2, 0, False))

      return transitions

    raise ValueError('Invalid action')

  def reset(self):
    self.state = self.start_state
    return self.state

  def step(self, action):
    
    # Use the transition model to get the possible next states, rewards, and done flags for the given action
    transitions = self.transition_model(self.state, action)

    # Sample one of the possible transitions based on their probabilities
    probs = [t[0] for t in transitions]

    # 랜덤하게 다음 상태, 보상, 종료 여부를 선택 -> 확률에 따라 선택 
    idx = np.random.choice(len(transitions), p=probs)
    _, next_state, reward, done = transitions[idx]

    self.state = next_state #Update the current state
    return next_state, reward, done


def equiprobable_random_policy(env):    
  return np.random.choice(env.action_space, p=[0.5, 0.5]) #equiprobable random action selection


def equiprobable_policy_prob(action):
  if action in ['draw', 'stop']:
    return 0.5
  return 0.0


def bellman_policy_evaluation(env, gamma=1.0, theta=1e-10):
  """
  Iterative policy evaluation for v_pi(s) using Bellman expectation equation.
  Policy: equiprobable random policy over {'draw', 'stop'}.
  """
  V = {s: 0.0 for s in env.non_terminal_states}

  while True:
    delta = 0.0

    for s in env.non_terminal_states:
      old_v = V[s]
      new_v = 0.0

      for action in env.action_space:
        # π(a|s) = 0.5 for both 'draw' and 'stop' actions
        pi_a = equiprobable_policy_prob(action)
        transitions = env.transition_model(s, action)

        expected_return = 0.0
        for prob, next_state, reward, done in transitions:
          next_v = 0.0 if done else V[next_state]
          expected_return += prob * (reward + gamma * next_v)

        new_v += pi_a * expected_return

      V[s] = new_v
      delta = max(delta, abs(old_v - new_v))

    if delta < theta:
      break

  return V



def main():
  alpha = 0.5 #probability of drawing a card with value 2
  env = Cardgame(alpha)
  discount_rate = 1.0
  num_eps = 10

  for i in range(num_eps):
    print('Episode {}/{}'.format(i+1, num_eps))

    total_reward = 0
    state = env.reset() #reset the environment to start a new episode
    done = False
    print('Initial state:', state)

    t = 0
    while not done:
      action = equiprobable_random_policy(env) #select an action using the equiprobable random policy
      next_state, reward, done = env.step(action) #take a step in the environment
      
      total_reward += (discount_rate ** t) * reward #accumulate reward
      t += 1

      print('Action:', action, '| Next state:', next_state, '| Reward:', reward)
    
    print('Total reward for episode {}: {}'.format(i+1, total_reward))
    print('---')

  # Step 2: Bellman expectation policy evaluation (for equiprobable policy)
  V_pi = bellman_policy_evaluation(env, gamma=discount_rate)
  print('\nBellman policy evaluation result (v_pi):')
  for s in env.non_terminal_states:
    print('v_pi({}) = {:.6f}'.format(s, V_pi[s]))


if __name__ == "__main__":  main()
