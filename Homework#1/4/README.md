# Mini Blackjack MDP

This assignment models a simplified card game as a **Markov Decision Process (MDP)** and generates 10 episodes using an equiprobable random policy.

## 1) Theory: MDP and Policy
This environment satisfies the Markov property because the next state depends only on the current state and action, not on the past history.

An MDP is defined by the following tuple:

$$
\mathcal{M} = (S, A, P, R, \gamma)
$$

- $S$: set of states
- $A$: set of actions
- $P(s' \mid s, a)$: probability of transitioning to state $s'$ after taking action $a$ in state $s$
- $r(s, a)$: immediate reward received after taking action $a$ in state $s$
- $\gamma$: discount factor

A policy $\pi$ maps states to actions. A deterministic policy can be written as:

$$
A_t = \pi(s_t)
$$

The cumulative reward (return) is defined as:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots
= \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$

The agent's objective is to find a policy that maximizes the expected return, $\mathbb{E}[G_t]$.

### Episodic Task and Terminal States

This problem is an **episodic task** with terminal states. The process terminates when a terminal state (e.g., $T_{STOP}$ or $T_{BUST}$) is reached. Consequently, the time horizon $t$ is finite, and the return $G_t$ is a finite sum rather than an infinite series:

$$
G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}
$$

where $T$ is the episode termination time.

### State-Value Function

Under a policy $\pi$, the value of state $s$ is defined as the expected return starting from that state:

$$
v_\pi(s) = \mathbb{E}_\pi [G_t \mid S_t = s]
$$

In practice, $v_\pi(s)$ is typically estimated from collected episode data. The episodes generated in this assignment provide the raw material for such estimation.

## 2) Assignment Setup

The goal of this assignment is to map the above concepts to a card-game environment, implement the MDP, and observe interaction outcomes under a fixed policy.

### Environment

- **Initial state:** $s_0 = 0$
- **Actions:** `draw`, `stop`
- **States:** $\{0, 1, 2, 3, 4, 5, T_{STOP}, T_{BUST}\}$
- **Discount factor:** $\gamma = 1$

### Transition Model

When `draw` is selected, the card value is either 1 or 2:

- $P(\text{card}=2) = \alpha$
- $P(\text{card}=1) = 1-\alpha$

The next state is determined by adding the card value to the current sum. The state transition and reward are jointly determined by $p(s', r \mid s, a)$:

**For `draw` action:**

$$
p(s+2, 0 \mid s, \text{draw}) = \alpha \quad \text{if } s+2 \leq 5
$$

$$
p(s+1, 0 \mid s, \text{draw}) = 1-\alpha \quad \text{if } s+1 \leq 5
$$

$$
p(T_{\text{BUST}}, -1 \mid s, \text{draw}) = \alpha + (1-\alpha) = 1 \quad \text{if } s+1 > 5 \text{ or } s+2 > 5
$$

**For `stop` action:**

$$
p(T_{\text{STOP}}, r(s,\text{stop}) \mid s, \text{stop}) = 1 \quad \text{for any } s
$$

```python
# This code is implemented P(s', r | s, a) for the MDP environment
if action == 'draw':
    if np.random.rand() < alpha:
        next_state = state + 2
    else:
        next_state = state + 1

    if next_state > 5:
        next_state = 'T_BUST'
        reward = -1
    else:
        reward = 0

elif action == 'stop':
    next_state = 'T_STOP'
    reward = r(state, 'stop')
```

where $r(s, \text{stop})$ is defined in the reward model below.

### Reward Model

- `stop`
	- $r(s, stop)=0$ for $s \in \{0,1,2,3\}$
	- $r(4, stop)=1$
	- $r(5, stop)=2$
- `draw`
  - Reward is 0 if the next state is non-bust
  - Reward is -1 if bust occurs

## 3) Policy Used in This Assignment

For all non-terminal states, the following equiprobable random policy is used:

$$
\pi(a \mid s) =
\begin{cases}
0.5, & a=\text{draw} \\
0.5, & a=\text{stop}
\end{cases}
$$

That is, the agent selects `draw` and `stop` with equal probability, regardless of the current state.

```python
# Define the action space
self.action_space = ['draw', 'stop']  

# ...

# Equiprobable random policy
def equiprobable_random_policy(env):    
  return np.random.choice(env.action_space, p=[0.5, 0.5]) # equiprobable random action selection
```

## 4) Episode Results (10 runs)

| Episode | Main Trajectory | Terminal State | Return |
| --- | --- | --- | --- |
| 1 | 0 → 1 → Stop | $T_{STOP}$ | 0.0 |
| 2 | 0 → 1 → 3 → 5 → Draw(Bust) | $T_{BUST}$ | -1.0 |
| 3, 4, 5, 9 | 0 → Stop | $T_{STOP}$ | 0.0 |
| 6, 7 | 0 → 2 → Stop | $T_{STOP}$ | 0.0 |
| 8 | 0 → 1 → Stop | $T_{STOP}$ | 0.0 |
| 10 | 0 → 1 → 3 → 4 → Stop | $T_{STOP}$ | 1.0 |

## 5) Interpretation and Connection to Value Estimation

Because the policy is random and does not use state value information, irrational choices can occur even in favorable states. For example, at state 5, `stop` is usually preferable (returning reward 2), but an equiprobable policy can still choose `draw` with 50% probability, risking a bust and penalty of -1.

Observe in Episode 2 the trajectory: $0 \to 1 \to 3 \to 5 \to \text{Draw(Bust)}$, resulting in a return of -1. In contrast, Episode 10 reaches state 4 and chooses `stop`, obtaining a return of 1.

**Theoretical Connection:** The 10 episodes collected above provide trajectory data. These episodes record actual returns $G_t$ for each starting state. To estimate $v_\pi(s)$ from such data, one would apply a method like first-visit Monte Carlo (FVMC):

$$
v_\pi(s) \approx \frac{1}{N(s)} \sum_{i \in \{\text{episodes where } S_t = s\}} G_t^{(i)}
$$

where $N(s)$ is the count of episodes visiting state $s$.

**Current Assignment Scope:** This assignment focuses on (i) implementing the MDP, (ii) executing 10 episodes under a fixed policy, and (iii) reporting trajectories and returns. Value function estimation is deferred to a future step.

>지금 이 과제에서 구현된 MDP는 equiporbable random policy를 사용했다. 즉, 행동 선택이 상태에 의존하지 않고 무작위로 이루어진다. 따라서, 에이전트는 상태의 가치를 고려하지 않고 행동을 선택한다. 이로 인해, 때때로 최적 행동이 아닌 행동을 선택할 수 있으며, 이는 에피소드 2에서 볼 수 있다. 반면에, 에피소드 10에서는 운 좋게도 최적 행동인 `stop`이 선택되어 좋은 결과를 얻었다. 이러한 결과는 에이전트가 상태 가치에 대한 정보를 활용하지 않고 행동을 선택할 때 발생할 수 있는 비합리적인 행동의 예시를 보여준다. 향후 과제에서는 이러한 데이터를 활용하여 상태 가치 추정 방법을 적용할 것이다.

For extensions including value function estimation, policy evaluation, and advanced learning strategies, see [Homework #4+](../4+/).