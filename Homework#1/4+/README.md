# Card Game MDP (4+) - Bellman Evaluation
This document explains the implementation of a card game MDP in `4+.py` and how it relates to the theoretical concepts of MDPs, policies, and Bellman evaluation. The structure is:
1. Theory first: what the Bellman equation means
2. Code mapping: where each concept is implemented
3. Interpretation: what the computed values tell us

---

## 1) Theoretical Background

### 1.1 MDP Definition

The card game is modeled as a Markov Decision Process (MDP):

$$
\mathcal{M} = (S, A, P, R, \gamma)
$$

where:

- $S$ is the state space
- $A$ is the action space
- $P(s', r \mid s, a)$ is the joint transition-reward probability
- $R$ is the reward mechanism
- $\gamma$ is the discount factor

In this game:

- States: $\{0,1,2,3,4,5,T_{STOP},T_{BUST}\}$
- Actions: `draw`, `stop`
- Discount factor: $\gamma = 1$

The important point is that an MDP does not only describe "what state comes next," but also "what reward is received together with that transition." That is why the full probabilistic object is written as $p(s', r \mid s, a)$.

### 1.2 Episodic Task

This problem is an episodic task because every run eventually ends in a terminal state:

- $T_{STOP}$ if the agent chooses `stop`
- $T_{BUST}$ if the sum exceeds 5 after `draw`

Because the episode terminates, the return is a finite sum:

$$
G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}
$$

where $T$ is the terminal time step.

This is different from continuing tasks, where the return may involve an infinite horizon. Here, the episode ends, so the sum stops at termination.

### 1.3 State-Value Function

For a fixed policy $\pi$, the state-value function is defined as:

$$
v_\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s]
$$

This means:

- start from state $s$
- follow policy $\pi$
- measure the return $G_t$
- average that return over all possible future outcomes

So $v_\pi(s)$ is not the reward for one episode. It is the **expected long-term return** from state $s$ when the agent follows policy $\pi$.

### 1.4 Bellman Expectation Equation

The Bellman expectation equation is the recursive definition of $v_\pi(s)$:

$$
v_\pi(s) = \sum_a \pi(a\mid s) \sum_{s',r} p(s',r\mid s,a)\left[r + \gamma v_\pi(s')\right]
$$

This equation says:

1. choose an action $a$ according to the policy $\pi(a\mid s)$
2. after that action, the environment may move to several possible next states $s'$ with different rewards $r$
3. for each possible outcome, add the immediate reward $r$ and the discounted value of the next state $\gamma v_\pi(s')$
4. average all of these possibilities using the probabilities

This is the core idea behind Bellman evaluation: the value of the current state is defined by the values of the next states.

### 1.5 Why Bellman Equation Matters

Bellman evaluation lets us compute state values **without waiting for countless full episodes**. Instead of only observing samples, we can use the model of the environment to estimate expected values directly.

This is why the transition model must be explicit in the code: Bellman updates need all possible next-state/reward outcomes, not just one sampled outcome.

In this assignment, the policy is equiprobable random:

$$
\pi(\text{draw}\mid s)=\pi(\text{stop}\mid s)=0.5
$$

---

## 2) Where Theory Is Applied in Code

### 2.1 Transition Dynamics: $p(s', r \mid s, a)$

Implemented in `transition_model(state, action)`.

This function returns a list of possible outcomes in the form:

```python
(probability, next_state, reward, done)
```

This corresponds exactly to the MDP joint dynamics $p(s', r \mid s, a)$.

For example:

- if `action == 'stop'`, the next state is always `T_STOP`
- if `action == 'draw'`, the next state depends on whether the card is 1 or 2

This separation is important because Bellman evaluation needs to consider **all** possible outcomes, not just one sampled step.

### 2.2 Sampling from the Environment

Implemented in `step(action)`.

This function does not compute the full expectation. Instead, it:

1. calls `transition_model(...)`
2. samples one outcome according to the transition probabilities
3. returns the chosen next state, reward, and termination flag

So `step()` is the **generative simulation** version of the environment, while `transition_model()` is the **analytic** version used for Bellman computation.

### 2.3 Policy Definition: $\pi(a\mid s)$

Implemented in `equiprobable_policy_prob(action)`.

The policy used in the Bellman evaluation is:

```python
0.5 for draw
0.5 for stop
```

This means the agent does not prefer one action over the other. The policy is fixed and random.

### 2.4 Bellman Policy Evaluation

Implemented in `bellman_policy_evaluation(env, gamma, theta)`.

This function:

- initializes all non-terminal state values to 0
- repeatedly applies the Bellman expectation equation
- uses the current estimate of $V(s')$ when computing $V(s)$
- stops when the maximum change across states becomes smaller than $\theta$

This is the iterative form of policy evaluation.

The update is a form of **bootstrapping** because the value of a state is estimated using the current estimate of the next state's value.

---

## 3) What the Results Mean

Example Bellman evaluation result with $\alpha=0.5$ and $\gamma=1$:

```text
v_pi(0) = 0.060059
v_pi(1) = 0.091797
v_pi(2) = 0.148438
v_pi(3) = 0.218750
v_pi(4) = 0.375000
v_pi(5) = 0.500000
```

### Interpretation

1. Under the current random policy, state 5 has the highest expected return.
2. This does **not** mean the agent always receives reward 2 at state 5.
3. It means that, on average, the long-term return from state 5 is better than from the other states under the same policy.

**Why is $v_\pi(5)$ the largest?**

- At state 5, `stop` gives reward 2.
- But the random policy still chooses `draw` with probability 0.5.
- If the agent draws, it busts and gets reward -1.
- Therefore the expected return is:

$$
v_\pi(5) = 0.5 \cdot 2 + 0.5 \cdot (-1) = 0.5
$$

This explains why the value is high, but not equal to 2.

### Why is $v_\pi(4)$ lower than you might expect?

At state 4:

- `stop` gives reward 1
- `draw` may move to state 5 or bust
- because the policy is random, the bad branch reduces the expected value

So the value is not only determined by the immediate best action, but by the **policy actually being followed**.

That is the main lesson of Bellman evaluation: value depends on both the environment and the policy.

---

## 4) Conclusion

This version connects simulation and value reasoning in one workflow.

- Episode rollout shows what happened in sampled interactions.
- Bellman policy evaluation shows what is expected under the policy.
- The transition model makes the environment mathematically explicit.

In short, this upgrade turns the original simulation into a real RL study example:

1. define the MDP
2. evaluate a policy using Bellman expectation updates
3. interpret the resulting values as long-term expected returns

That is the core bridge from random interaction to value-based reasoning.
