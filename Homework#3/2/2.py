import numpy as np

class DroneDelivery:
  def __init__(self):

      #(a)-1
      self.height = 6
      self.width = 6
      self.max_battery = 20


      #(a)-2
      self.state_space =  [(x, y, b)
                    for x in range(self.height)
                    for y in range(self.width)
                    for b in range(self.max_battery + 1)]
      self.action_space =  [0, 1, 2, 3]  # Up, Down, Left, Right


      #(a)-3
      self.obstacles = [(1,1), (1,2), (3,1), (4,3)]
      self.windy = [(0,5), (1,5), (3,2), (3,3), (3,4)]
      self.state_space = [(x, y, b) for (x, y, b) in self.state_space if (x, y) not in self.obstacles]

      self.start_grid = (0, 0)
      self.goal_grid = (5, 5)

      self.state = self.start_grid + (self.max_battery,)

  def reset(self):
    self.state = self.start_grid + (self.max_battery, )
    return self.state

  def step(self, action):
    #(a)-4 (x, y): current grid, b: current battery level
    x = self.state[0]
    y = self.state[1]
    b = self.state[2]


    #(a)-5 Battery consumption: -1 for normal grids, -3 for windy grids
    if (x, y) in self.windy:
        b -= 3
    else:
        b -= 1


    #(a)-6 Return the battery exhaustion failure before reaching the goal
    if b < 0:
      self.state = (x, y, 0)
      return (x, y, 0), -50, True


    #(a)-7 Grid movement (deterministic move)
    prev_x, prev_y = x, y   # 장애물 복원용
    if action == 0:
         x -= 1
    elif action == 1:
         x += 1
    elif action == 2:
         y -= 1
    elif action == 3:
         y += 1
         
    # 격자 경계 처리 (경계 밖이면 원래 위치로)
    if x < 0 or x >= self.height or y < 0 or y >= self.width:
        x, y = prev_x, prev_y


    #(a)-8 Check whether the next state is obstacle: undo the movement
    if (x, y) in self.obstacles:
      x = prev_x
      y = prev_y


    #(a)-9 Reaching the goal
    if (x, y) == self.goal_grid:
      self.state = (x, y, b)
      return (x, y, b), 100, True


    #(a)-10 Normal moves
    else:
      self.state = (x, y, b)
      return (x, y, b), -1, False


def argmax(arr):
    return np.random.choice([idx for idx in range(len(arr)) if arr[idx] == arr.max()])

#SARSA code (5 pts)
'''

Write down the SARSA code here

'''
class SARSA:
  def __init__(self, env, gamma = 1.0, epsilon = 0.2, alpha = 0.01):
    self.env = env
    self.gamma = gamma
    self.epsilon = epsilon
    self.alpha = alpha
    self.q_table = {state:np.zeros(len(self.env.action_space)) for state in self.env.state_space}

  def epsilon_greedy(self, state):
    if np.random.rand() < self.epsilon:
      return np.random.choice(self.env.action_space)
    else:
      return argmax(self.q_table[state])

  def control(self, episodes = 200):
    cumulative_steps = []
    total_steps = 0

    for i in range(1, episodes+1):
      if i % int(episodes/10) == 0:
        print('Episode ', i, '/', episodes, ': ', '[', '*'*int(i/(episodes/10)), '-'*int((episodes - i)/(episodes/10)), ']')

      state = self.env.reset()
      action = self.epsilon_greedy(state)
      done = False

      while (not done):
        next_state, reward, done = self.env.step(action)
        next_action = self.epsilon_greedy(next_state)
        self.q_table[state][action] += self.alpha*(reward + self.gamma*self.q_table[next_state][next_action] - self.q_table[state][action])
        state = next_state
        action = next_action
        total_steps += 1

      cumulative_steps.append(total_steps)

    policy = {state: argmax(actions) for state, actions in self.q_table.items()}
    return policy, self.q_table, cumulative_steps


#Q-Learning code (5 pts)
'''

Write down the Q-Learning code here

'''
class QLearning:
  def __init__(self, env, gamma = 1.0, epsilon = 0.2, alpha = 0.01):
    self.env = env
    self.gamma = gamma
    self.epsilon = epsilon
    self.alpha = alpha
    self.q_table = {state:np.zeros(len(self.env.action_space)) for state in self.env.state_space}

  def epsilon_greedy(self, state):
    if np.random.rand() < self.epsilon:
      return np.random.choice(self.env.action_space)
    else:
      return argmax(self.q_table[state])

  def control(self, episodes = 200):
    cumulative_steps = []
    total_steps = 0

    for i in range(1, episodes+1):
      if i % int(episodes/10) == 0:
        print('Episode ', i, '/', episodes, ': ', '[', '*'*int(i/(episodes/10)), '-'*int((episodes - i)/(episodes/10)), ']')

      state = self.env.reset()
      done = False

      while (not done):
        action = self.epsilon_greedy(state)
        next_state, reward, done = self.env.step(action)
        self.q_table[state][action] += self.alpha*(reward + self.gamma*max(self.q_table[next_state]) - self.q_table[state][action])
        state = next_state
        total_steps += 1

      cumulative_steps.append(total_steps)

    policy = {state: argmax(actions) for state, actions in self.q_table.items()}
    return policy, self.q_table, cumulative_steps
  

#Double Q-Learning code (10 pts)
'''
    
Write down the Double Q-Learning code here

'''
class DoubleQLearning:
  def __init__(self, env, gamma = 1.0, epsilon = 0.2, alpha = 0.01):
    self.env = env
    self.gamma = gamma
    self.epsilon = epsilon
    self.alpha = alpha

    # Two separate Q-tables for Double Q-Learning
    self.q1_table = {state: np.zeros(len(self.env.action_space)) for state in self.env.state_space}
    self.q2_table = {state: np.zeros(len(self.env.action_space)) for state in self.env.state_space}

  def epsilon_greedy(self, state):
    if np.random.rand() < self.epsilon:
      return np.random.choice(self.env.action_space)
    else:

      # Select action based on the combined Q-values from both tables
      return argmax(self.q1_table[state] + self.q2_table[state])

  def control(self, episodes=200):
    cumulative_steps = []
    total_steps = 0

    for i in range(1, episodes+1):
      if i % int(episodes/10) == 0:
        print('Episode ', i, '/', episodes, ': ', '[', '*'*int(i/(episodes/10)), '-'*int((episodes-i)/(episodes/10)), ']')

      state = self.env.reset()
      done = False

      while not done:
        action = self.epsilon_greedy(state)
        next_state, reward, done = self.env.step(action)

        if np.random.rand() < 0.5:
          # Update Q1 using the action that maximizes Q1, but evaluate using Q2
          best_a = argmax(self.q1_table[next_state])
          self.q1_table[state][action] += self.alpha * (
            reward + self.gamma * self.q2_table[next_state][best_a] - self.q1_table[state][action]
          )
        else:
          # Update Q2 using the action that maximizes Q2, but evaluate using Q1
          best_a = argmax(self.q2_table[next_state])
          self.q2_table[state][action] += self.alpha * (
            reward + self.gamma * self.q1_table[next_state][best_a] - self.q2_table[state][action]
          )

        state = next_state
        total_steps += 1

      cumulative_steps.append(total_steps)

    q_combined = {state: self.q1_table[state] + self.q2_table[state] for state in self.env.state_space}
    policy = {state: argmax(actions) for state, actions in q_combined.items()}
    return policy, q_combined, cumulative_steps
  

#Training & Find the optimal path of the 3 algorithms (5 pts)
env = DroneDelivery()
sarsa_opt_policy, _, sarsa_steps = SARSA(env).control(episodes=10000)
ql_opt_policy,    _, ql_steps    = QLearning(env).control(episodes=10000)
dql_opt_policy,   _, dql_steps   = DoubleQLearning(env).control(episodes=10000)

def print_optimal_path(policy, env, title=''):
    print(f'\n=== {title} Optimal Path ===')
    grid_map = [[' . ' for _ in range(env.width)] for _ in range(env.height)]
    for ox, oy in env.obstacles: grid_map[ox][oy] = ' O '
    for wx, wy in env.windy:     grid_map[wx][wy] = ' W '
    grid_map[env.start_grid[0]][env.start_grid[1]] = ' S '
    grid_map[env.goal_grid[0]][env.goal_grid[1]]   = ' G '
    state, done, arrows = env.reset(), False, {0:' ^ ', 1:' v ', 2:' < ', 3:' > '}
    while not done:
        action = policy[state]; x, y, _ = state
        if (x, y) not in (env.start_grid, env.goal_grid): grid_map[x][y] = arrows[action]
        state, reward, done = env.step(action)
        if reward == -50: print("[Warning] Battery Exhausted."); break
    sep = '+---+---+---+---+---+---+'
    print(sep)
    for row in grid_map:
        print('|' + '|'.join(row) + '|'); print(sep)

def show_sarsa(): print_optimal_path(sarsa_opt_policy, env, 'SARSA')
def show_ql():    print_optimal_path(ql_opt_policy,    env, 'Q-Learning')
def show_dql():   print_optimal_path(dql_opt_policy,   env, 'Double Q-Learning')

show_sarsa()
show_ql()
show_dql()


#Draw the comparison figure (5 pts)
import matplotlib.pyplot as plt

episodes = list(range(1, 10001))

plt.figure(figsize=(10, 6))
plt.plot(sarsa_steps, episodes, label='SARSA', color='blue')
plt.plot(ql_steps,    episodes, label='Q-Learning', color='green')
plt.plot(dql_steps,   episodes, label='Double Q-Learning', color='red')
plt.xlabel('Cumulative Time Steps')
plt.ylabel('Episodes')
plt.title('Episodes vs Cumulative Time Steps\n(SARSA vs Q-Learning vs Double Q-Learning)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('comparison.png', dpi=150)
plt.show()