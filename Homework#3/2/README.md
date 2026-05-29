# Temporal Difference Learning
Temporal Difference Learning (TD Learning) is a reinforcement learning method that combines ideas from Monte Carlo methods and dynamic programming. It allows an agent to learn from incomplete episodes by updating value estimates based on the difference between predicted and actual rewards.

TD Learning updates the value function $V(s)$ for a state $s$ using the following formula:

$$ V(s) \leftarrow V(s) + \alpha [R_{t+1} + \gamma V(s_{t+1}) - V(s)] $$

Where:
- $\alpha$ is the learning rate, which determines how much the value function is updated based on new information.
- $R_{t+1}$ is the reward received after taking an action in state $s$ and transitioning to state $s_{t+1}$.
- $\gamma$ is the discount factor, which determines the importance of future rewards.

## SARSA Learning
SARSA (State-Action-Reward-State-Action) is an on-policy TD control algorithm that learns the action-value function $Q(s, a)$ for a given policy. The update rule for SARSA is as follows:

$$ Q(s, a) \leftarrow Q(s, a) + \alpha [R_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s, a)] $$

Where:
- $Q(s, a)$ is the estimated action-value for taking action $a$ in state $s$.
- $a_{t+1}$ is the action taken in the next state $s_{t+1}$ according to the current policy.

The agent updates its action-value estimates based on the action taken in the next state, which is why SARSA is considered an on-policy algorithm.

## Q-Learning
Q-Learning is an off-policy TD control algorithm that learns the optimal action-value function $Q^*(s, a)$ regardless of the policy being followed. The update rule for Q-Learning is as follows:

$$ Q(s, a) \leftarrow Q(s, a) + \alpha [R_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s, a)] $$

Where:
- $Q(s, a)$ is the estimated action-value for taking action $a$ in state $s$.
- $\max_{a'} Q(s_{t+1}, a')$ is the maximum action-value for the next state $s_{t+1}$.

The agent updates its action-value estimates based on the maximum possible reward in the next state, which is why Q-Learning is considered an off-policy algorithm.

## Double Q-Learning
Double Q-Learning is an extension of Q-Learning that addresses the overestimation bias in action-value estimates. It maintains two separate action-value functions, $Q_1$ and $Q_2$, and updates them alternately. The update rules for Double Q-Learning are as follows:

$$ Q_1(s, a) \leftarrow Q_1(s, a) + \alpha [R_{t+1} + \gamma Q_2(s_{t+1}, \arg\max_{a'} Q_1(s_{t+1}, a')) - Q_1(s, a)] $$
$$ Q_2(s, a) \leftarrow Q_2(s, a) + \alpha [R_{t+1} + \gamma Q_1(s_{t+1}, \arg\max_{a'} Q_2(s_{t+1}, a')) - Q_2(s, a)] $$

Where:
- $Q_1(s, a)$ and $Q_2(s, a)$ are the two separate action-value estimates for taking action $a$ in state $s$.
- The action selection for the next state is based on the maximum action-value from the other Q-function, which helps to reduce overestimation bias.

# Drone Delivery Environment
In this homework, we will implement a drone delivery environment.


![alt text](./drone_delivery_env.png)   
**grid size: 6x6**

**state space:** (x, y, b) where x and y are the coordinates of the drone and b is the remaining battery level

$$(x, y, b) \quad \text{where} \quad x \in \{0, 1, 2, 3, 4, 5\}, \quad y \in \{0, 1, 2, 3, 4, 5\}, \quad b \in \{0, 1, ..., 20\}$$

**action space:** up, down, left, right

$$A = \{0: \text{up}, ~1: \text{down}, ~2: \text{left}, ~3: \text{right}\}$$

**reward:** 1 for each step, -3 when the drone leaves the windy area and can't go through an obstacle, -50 when the drone's battery is depleted, and +100 when the drone reaches the goal.

$$ R(s, a) = \begin{cases}
-1 & \text{for each step} \\
-3 & \text{when the drone leaves the windy area} \\
-50 & \text{when the drone's battery is depleted} \\
+100 & \text{when the drone reaches the goal}
\end{cases} $$

**state transition:** The state transition is deterministic, meaning that the drone will always move in the intended direction unless it encounters an obstacle or leaves the windy area. The transition probabilities can be defined as follows:

$$ P(s'|s, a) = \begin{cases}
1 & \text{if the action leads to a valid state transition} \\
0 & \text{otherwise}
\end{cases} $$

## Learning Optimal Policy
Train the agent using SARSA, Q-Learning, and Double Q-Learning algorithms to find the optimal policy for navigating the drone from the start position to the goal while avoiding obstacles and managing battery life effectively. 


We use the following hyperparameters for training:
$$\epsilon = 0.2, \quad \alpha = 0.01, \quad \text{episodes} = 10,000, \quad  Battery Level = 20 $$


After training, visualize the optimal paths learned by each algorithm on the grid.

```
=== SARSA Optimal Path ===
+---+---+---+---+---+---+
| S | . | . | . | . | W |
+---+---+---+---+---+---+
| v | O | O | . | . | W |
+---+---+---+---+---+---+
| > | > | v | . | . | . |
+---+---+---+---+---+---+
| . | O | > | > | v | . |
+---+---+---+---+---+---+
| . | . | . | O | v | . |
+---+---+---+---+---+---+
| . | . | . | . | > | G |
+---+---+---+---+---+---+

=== Q-Learning Optimal Path ===
+---+---+---+---+---+---+
| S | . | . | . | . | W |
+---+---+---+---+---+---+
| v | O | O | . | . | W |
+---+---+---+---+---+---+
| > | > | v | . | . | . |
+---+---+---+---+---+---+
| . | O | > | > | > | v |
+---+---+---+---+---+---+
| . | . | . | O | . | v |
+---+---+---+---+---+---+
| . | . | . | . | . | G |
+---+---+---+---+---+---+

=== Double Q-Learning Optimal Path ===
[Warning] Battery Exhausted.
+---+---+---+---+---+---+
| S | > | > | > | > | > |
+---+---+---+---+---+---+
| . | O | O | . | . | W |
+---+---+---+---+---+---+
| . | . | . | . | . | . |
+---+---+---+---+---+---+
| . | O | W | W | W | . |
+---+---+---+---+---+---+
| . | . | . | O | . | . |
+---+---+---+---+---+---+
| . | . | . | . | . | G |
+---+---+---+---+---+---+
```

This is the graph of cumulative timesteps for each algorithm:   
![alt text](1000000_b2000.png)