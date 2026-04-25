# Bellman Equation for Policy Evaluation
In this homework, we will implement the Bellman equation for policy evaluation. The Bellman equation is a fundamental concept in reinforcement learning that describes the relationship between the value of a state and the values of its next states.

## Bellman Equation
The Bellman equation for policy evaluation can be expressed as follows:
$$ v_{\pi}(s) = \sum_{a} \pi(a|s) \sum_{s', r} p(s', r|s, a) [r + \gamma v_{\pi}(s')]$$

Where:
- $v_{\pi}(s)$ is the value of state s under policy $\pi$.
- $r(s, a)$ is the reward received when taking action a in state s.
- $γ$ is the discount factor, which determines the importance of future rewards.
- $p(s', r|s, a)$ is the probability of transitioning to state s' and receiving reward r from state s by taking action a.
- $\pi(a|s)$ is the probability of taking action a in state s under policy $\pi$.

```python
for action_idx, action_prob in enumerate(policy[state]):
    for trans_prob, next_state, reward in env.transition_prob(state, action_idx):
        v_new += action_prob * trans_prob * (reward + env.gamma * V[next_state]) #Bellman update rule
```


# Frozen Lake Environment
The Frozen Lake environment is a grid world where the agent must navigate from a starting point to a goal while avoiding holes. The environment is represented as a grid, and the agent can take actions to move in four directions: up, down, left, and right.

The agent receives a reward of +2 for reaching the goal, a reward of -2 for falling into a hole, and a reward of 0 for all other transitions. The environment is stochastic, meaning that the agent's actions may not always lead to the intended outcome.

<img src="frozen_lake_env.png" alt="alt text" width="65%">

## Implementation
To implement the Bellman equation for policy evaluation, we will follow these steps:
1. Define the environment and the policy.
2. Initialize the value function for all states. (e.g., set all values to zero).
3. Iteratively update the value function using the Bellman equation until convergence.
4. Output the final value function for all states.

```
Total Iteration: 28
Value Function (V_pi) for Random Policy:
[[-1.99 -1.98  0.   -1.97]
 [ 0.   -1.96 -1.86 -1.94]
 [-1.86  0.   -1.53  0.  ]
 [-1.57 -1.28 -0.27  0.  ]]
 ```

## Conclusion
In this homework, we have implemented the Bellman equation for policy evaluation in the context of the Frozen Lake environment. By iteratively applying the Bellman update rule, we were able to compute the value function for a random policy. This value function provides insights into the expected returns for each state under the given policy, which can be used to improve the policy in future iterations.