import numpy as np
import matplotlib.pyplot as plt

#====================================================================
# Environment: Online Advertisement
#====================================================================
class OnlineAd:
  def __init__(self):
    self.size = 5                                       # number of bandits
    self.q =  np.array([0.3, 0.25, 0.4, 0.45, 0.35])    # true q values of each bandit
    
  def reward(self, action):
    # return 1 with the probability of q[action], otherwise return 0
    return 1 if np.random.rand() < self.q[action] else 0 
  
def argmax(arr):
  return np.random.choice([idx for idx in range(len(arr)) if arr[idx] == arr.max()])


# ====================================================================
# 1. epsilon-greedy action selection rule (with epsilon=0.1)
# ====================================================================
def epsilon_greedy(env, n, epsilon): #(bandit environment, the number of trials, epsilon)

  # history of the selected actions
  hist_A = [] 

  Q = np.zeros(env.size) # initial Q
  N = np.zeros(env.size) # selected times of each action

  for i in range(n):
    if np.random.rand() > epsilon:    # greedy action selection with the prob. (1-epsilon)
      A = argmax(Q)

    else:
      A = np.random.randint(env.size) # exploration with the prob. (epsilon)

    R = env.reward(A) # receive a reward

    # Update Q: incremental implementation
    N[A] += 1
    Q[A] += (1/N[A]) * (R - Q[A])

    # save the current action
    hist_A.append(A)

  # return the final Q and the history of selected actions
  return Q, np.array(hist_A) 


# =====================================================================
# Initialize the environment
# =====================================================================
env = OnlineAd()
n_trials = 2000
n_exp = 1000


# =====================================================================
# Training of epsilon-greedy action selection rule (with epsilon=0.1)
# =====================================================================
A_EGreedy = [] #experiments of epsilon-greedy

for i in range(1, n_exp+1):
  Q, hist_A = epsilon_greedy(env, n_trials, epsilon=0.1)
  A_EGreedy.append(hist_A)


  if i % 100 == 0 and i > 0:
    print('{}/{}'.format(i, n_exp), '[', '*'*int(i/100), '-'*int((n_exp-i)/100), ']')

A_EGreedy = np.array(A_EGreedy)


# =====================================================================
# 2. optimistic initial value (with epsilon=0.0 and init_value=10)
# =====================================================================
def opt_init(env, n, epsilon, init_value): # (bandit environment, the number of trials, epsilon)

  #record the history
  hist_A = [] #history of the selected actions

  Q = np.ones(env.size)*init_value #initial Q (Optimistic)
  N = np.zeros(env.size) #selected times of each action

  for i in range(n):
    if np.random.rand() > epsilon:  #exploitation with the prob. (1-epsilon)
      A = argmax(Q)

    else:
      A = np.random.randint(env.size) #exploration with the prob. (epsilon)

    R = env.reward(A) #receive a reward

    #Update Q: incremental implementation
    N[A] += 1
    Q[A] += (1/N[A]) * (R - Q[A])

    #save the current action
    hist_A.append(A)

  return Q, np.array(hist_A)


# =====================================================================
# Training of optimistic initial value (with epsilon=0.0 and init_value=10)
# =====================================================================
A_OPT = [] # eps=0, greedy, optimistic

for i in range(1, n_exp+1):
  Q, hist_A  = opt_init(env, n=n_trials, epsilon=0, init_value = 10)
  A_OPT.append(hist_A)

  if i % 100 == 0 and i > 0:
    print('{}/{}'.format(i, n_exp), '[', '*'*int(i/100), '-'*int((n_exp-i)/100), ']')

A_OPT = np.array(A_OPT)


# =====================================================================
# 3. upper confidence bound (with c=2)
# =====================================================================
def upper_confidence_bound(env, n, c_UCB): #(bandit environment, number of trials, confidence level)
  #record the history
  hist_A = []

  Q = np.zeros(env.size)
  N = np.zeros(env.size)

  N = N + 0.00001 #avoding zero-division
  #iteration
  for i in range(n):
      #upper confidence bound
      UCB = c_UCB*np.sqrt(np.log(n)/N)
      A = argmax(Q+UCB)
      R = env.reward(A)

      N[A] += 1
      #undo N treatment if it avoids zero
      if N[A] == 1.00001:
          N[A] -= 0.00001
      #incremental implementation
      Q[A] += (1/N[A]) * (R - Q[A])
      #save the current action
      hist_A.append(A)

  return Q, np.array(hist_A)


# =====================================================================
# Training of upper confidence bound (with c=2)
# =====================================================================
A_UCB = []     # upper-confidence-bound

for i in range(1, n_exp+1):
  Q, hist_A = upper_confidence_bound(env, n=n_trials, c_UCB = 2)
  A_UCB.append(hist_A)

  if i % 100 == 0 and i > 0:
    print('{}/{}'.format(i, n_exp), '[', '*'*int(i/100), '-'*int((n_exp-i)/100), ']')

A_UCB = np.array(A_UCB)


# =====================================================================
# Calculate the optimal action selection probability for each algorithm
# =====================================================================
max_A = argmax(env.q)

# Calculate boolean arrays indicating if the optimal action was selected
is_optimal_EGreedy = (A_EGreedy == max_A)
is_optimal_OPT = (A_OPT == max_A)
is_optimal_UCB = (A_UCB == max_A)

# Compute the average optimal action selection probability over experiments
mean_optimal_prob_EGreedy = np.average(is_optimal_EGreedy, axis=0)
mean_optimal_prob_OPT = np.average(is_optimal_OPT, axis=0)
mean_optimal_prob_UCB = np.average(is_optimal_UCB, axis=0)

print(mean_optimal_prob_EGreedy)
print(mean_optimal_prob_OPT)
print(mean_optimal_prob_UCB)


# =====================================================================
# Plot the average optimal action selection probability over steps for each algorithm
# =====================================================================
plt.figure(figsize=(12, 8))
plt.plot(mean_optimal_prob_EGreedy, color='green', label='epsilon-greedy (epsilon=0.1)')
plt.plot(mean_optimal_prob_OPT, color='red', label='optimistic initial value (epsilon=0.0)')
plt.plot(mean_optimal_prob_UCB, color='blue', label='UCB (c=2)')

plt.xlabel('Steps')
plt.ylabel('Optimal action selection probability')
plt.title('Average Optimal Action Selection Probability Over Steps')
plt.legend()
plt.grid(True)
plt.show()