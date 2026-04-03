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
    '''
    Define the state transition (return next_state, reward, done) (15 pts)
    (1) if draw => bust or not (10 pts)
    (2) if stop => non-bust terminal (5 pts)
    '''
    transitions = self.transition_model(self.state, action)

    probs = [t[0] for t in transitions]
    idx = np.random.choice(len(transitions), p=probs)
    _, next_state, reward, done = transitions[idx]

    self.state = next_state #Update the current state
    return next_state, reward, done


def equiprobable_random_policy(env):    
  return np.random.choice(env.action_space, p=[0.5, 0.5]) #equiprobable random action selection


def bellman_policy_evaluation(env, policy, gamma=1.0, theta=1e-10):
  """
  Iterative policy evaluation for v_pi(s) using Bellman expectation equation.
  Policy is a state-to-action mapping.
  """
  V = {s: 0.0 for s in env.non_terminal_states}

  while True:
    delta = 0.0

    for s in env.non_terminal_states:
      old_v = V[s]
      action = policy[s]
      transitions = env.transition_model(s, action)

      new_v = 0.0
      for prob, next_state, reward, done in transitions:
        next_v = 0.0 if done else V[next_state]
        new_v += prob * (reward + gamma * next_v)

      V[s] = new_v
      delta = max(delta, abs(old_v - new_v))

    if delta < theta:
      break

  return V


def action_value(env, state, action, V, gamma=1.0):
  q_value = 0.0
  transitions = env.transition_model(state, action)

  for prob, next_state, reward, done in transitions:
    next_v = 0.0 if done else V[next_state]
    q_value += prob * (reward + gamma * next_v)

  return q_value


def policy_iteration(env, gamma=1.0, theta=1e-10):
  policy = {s: 'stop' for s in env.non_terminal_states}
  iteration = 0

  while True:
    iteration += 1
    V = bellman_policy_evaluation(env, policy, gamma=gamma, theta=theta)

    new_policy = {}
    policy_stable = True

    for s in env.non_terminal_states:
      draw_value = action_value(env, s, 'draw', V, gamma)
      stop_value = action_value(env, s, 'stop', V, gamma)

      if draw_value > stop_value:
        best_action = 'draw'
      else:
        best_action = 'stop'

      new_policy[s] = best_action

      if best_action != policy[s]:
        policy_stable = False

    policy = new_policy

    print('\nPolicy iteration step {}'.format(iteration))
    for s in env.non_terminal_states:
      draw_value = action_value(env, s, 'draw', V, gamma)
      stop_value = action_value(env, s, 'stop', V, gamma)
      print('state {} | V = {:.6f} | Q(draw) = {:.6f} | Q(stop) = {:.6f} | policy = {}'.format(
        s, V[s], draw_value, stop_value, policy[s]
      ))

    if policy_stable:
      break

  return V, policy



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

  # Step 2: Bellman-based policy iteration
  V_star, optimal_policy = policy_iteration(env, gamma=discount_rate)

  print('\nFinal policy iteration result:')
  for s in env.non_terminal_states:
    draw_q = action_value(env, s, 'draw', V_star, discount_rate)
    stop_q = action_value(env, s, 'stop', V_star, discount_rate)
    print('state {} -> Q(draw) = {:.6f}, Q(stop) = {:.6f}, best action = {}'.format(
      s, draw_q, stop_q, optimal_policy[s]
    ))


if __name__ == "__main__":  main()
