import numpy as np

def argmax(arr):
  return np.random.choice([idx for idx in range(len(arr)) if arr[idx] == arr.max()])

class FrozenLake:
  def __init__(self):
    '''
    Define the environment
    '''
    self.height = 4
    self.width = 4
    self.state_space = [(x, y) for x in range(self.height) for y in range(self.width)]
    self.action_space = [0, 1, 2, 3] # Up, Down, Left, Right
    self.start_state = (0, 0)
    self.goal_state = (3, 3)
    self.holes = [(0, 2), (1, 0), (2, 1), (2, 3)]
    self.state = self.start_state

  def reset(self):
    self.state = self.start_state
    return self.state

  def step(self, action):
    '''
    Define the state transition
    '''
    x,y = self.state

    if action == 0:     # Up
      x = max(x-1, 0)

    elif action == 1:   # Down
      x = min(x+1, self.height-1)

    elif action == 2:   # Left
      y = max(y-1, 0)
    
    elif action == 3:   # Right
      y = min(y+1, self.width-1)

    self.state = (x, y)

    if self.state in self.holes:
        return self.state, -2, True  # next state, reward, done
    elif self.state == self.goal_state:
        return self.state, 2, True  # next state, reward, done
    else:
        return self.state, 0, False  # next state, reward, done

class MC_Control:
  def __init__(self, env, gamma = 1.0, epsilon = 0.2):
    '''
    Initialize the model
    '''
    self.env = env
    self.gamma = gamma
    self.epsilon = epsilon
    self.q_table = {state: np.zeros(len(self.env.action_space)) for state in self.env.state_space}
    self.returns = {state: {action: [] for action in self.env.action_space} for state in self.env.state_space}
    self.visits = {state: np.zeros(len(self.env.action_space)) for state in self.env.state_space}

  def epsilon_greedy(self, state):
    if np.random.rand() < self.epsilon:
      return np.random.choice(self.env.action_space)
    else:
      # If multiple actions have the same max Q-value, argmax will choose one randomly
      return argmax(self.q_table[state])

  def generate_episode(self):
    episode = []
    state = self.env.reset()
    done = False

    while (not done) and (len(episode) < 350):
      '''
      Generate an episode
      '''
      action = self.epsilon_greedy(state)
      next_state, reward, done = self.env.step(action)
      episode.append((state, action, reward))
      state = next_state
    return episode

  def update_q(self, episode):
    G = 0
    '''
    Update the action value based on the generated episode (first-visit)
    '''
    visited_state_actions = set()

    for t in reversed(range(len(episode))):
      state, action, reward = episode[t]
      G = reward + self.gamma * G

      if (state, action) not in visited_state_actions:
        self.returns[state][action].append(G)
        self.visits[state][action] += 1
        self.q_table[state][action] = np.mean(self.returns[state][action])
        visited_state_actions.add((state, action))
        

  def control(self, episodes=10000):
    for i in range(1, episodes+1):
      if i % int(episodes/10) == 0:
        print('Episode ', i, '/', episodes, ': ', '[', '*'*int(i/(episodes/10)), '-'*int((episodes - i)/(episodes/10)), ']')
      episode = self.generate_episode()
      self.update_q(episode)

    policy = {state: argmax(actions) for state, actions in self.q_table.items()}
    return policy, self.q_table

#Training
env = FrozenLake()

MC_FrozenLake = MC_Control(env)
opt_policy, q_table = MC_FrozenLake.control(10000)

#Visualize the optimal policy (do not change the code)
'''
Show the optimal policy
'''
policy = [[0 for _ in range(4)] for _ in range(4)]

for state, action in opt_policy.items():
  if action == 0:
    policy[state[0]][state[1]] = '^'
  elif action == 1:
    policy[state[0]][state[1]] = 'v'
  elif action == 2:
    policy[state[0]][state[1]] = '<'
  elif action == 3:
    policy[state[0]][state[1]] = '>'

policy[0][2], policy[1][0], policy[2][1], policy[2][3] = 'H', 'H', 'H', 'H'
policy[3][3] = 'G'
for row in policy:
    print(' '.join(str(cell) for cell in row))