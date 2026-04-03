import numpy as np

#4(a): 20 pts
class Cardgame:
  def __init__(self, alpha):
    '''
    Intialize the class: define state space, action space, initital state, alpha, etc. (5 pts)'''
    self.state_space = [0, 1, 2, 3, 4, 5, "T_STOP", "T_BUST"]   #define state space
    self.action_space = ['draw', 'stop']                        #define action space

    self.start_state = 0      #Start state
    self.state = 0            #Current state

    self.alpha = alpha        #alpha => +2 / 1-alpha => +1

  def reset(self):
    self.state = self.start_state
    return self.state

  def step(self, action):
    '''
    Define the state transition (return next_state, reward, done) (15 pts)
    (1) if draw => bust or not (10 pts)
    (2) if stop => non-bust terminal (5 pts)
    '''
    if action == 'draw':
      card = np.random.choice([1, 2], p=[1-self.alpha, self.alpha]) #Draw a card

      next_state = self.state + card #Update the state

      if next_state > 5: #bust
        reward = -1
        next_state = 'T_BUST' #terminal state
        done = True
      else: #not bust
        reward = 0
        done = False
      
    elif action == 'stop':
      next_state = 'T_STOP' #terminal state

      if self.state <= 3:
        reward = 0
      elif self.state == 4:
        reward = 1
      elif self.state == 5:
        reward = 2

      done = True

    self.state = next_state #Update the current state
    return next_state, reward, done


#4(b)
'''
define the equiprobable random policy (5 pts)
'''
def equiprobable_random_policy(env):    
  return np.random.choice(env.action_space, p=[0.5, 0.5]) #equiprobable random action selection


'''
generate 10 episodes and print the results (interaction sequence, cumulative reward, etc.) (5 pts)
'''
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


if __name__ == "__main__":  main()
