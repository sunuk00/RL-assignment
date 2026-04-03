# Card Game MDP (4++) - From Evaluation to Policy Update

This folder extends the `4+` version.

The key difference is:

- `4+`: Bellman **policy evaluation** only (measure how good a fixed policy is)
- `4++`: Bellman evaluation **plus policy improvement** (update policy using value estimates)

Main file: `4++.py`

---

## 1) Theory Background

### 1.1 MDP Definition

The environment is modeled as:

$$
\mathcal{M}=(S,A,P,R,\gamma)
$$

- $S$: states
- $A$: actions (`draw`, `stop`)
- $P(s',r\mid s,a)$: joint transition-reward dynamics
- $R$: reward structure
- $\gamma$: discount factor

States are:

$$
\{0,1,2,3,4,5,T_{STOP},T_{BUST}\}
$$

### 1.2 Bellman Evaluation vs. Policy Improvement

This is the most important concept in `4++`.

1. **Policy Evaluation** (Bellman expectation)
	- Given a policy $\pi$, compute $v_\pi(s)$:

$$
v_\pi(s)=\sum_{s',r} p(s',r\mid s,\pi(s))\left[r+\gamma v_\pi(s')\right]
$$

2. **Policy Improvement**
	- Use the evaluated value function to compare actions:

$$
q_\pi(s,a)=\sum_{s',r} p(s',r\mid s,a)\left[r+\gamma v_\pi(s')\right]
$$

	- Update policy greedily:

$$
\pi_{new}(s)=\arg\max_a q_\pi(s,a)
$$

3. Repeat until policy stops changing.

This loop is **Policy Iteration**.

---

## 2) How Theory Maps to `4++.py`

### 2.1 Environment Dynamics

Implemented in `transition_model(state, action)`.

It returns explicit outcomes:

```python
(probability, next_state, reward, done)
```

This is the direct code representation of $p(s',r\mid s,a)$.

### 2.2 Sample Interaction (Simulation)

Implemented in `step(action)`.

- Calls `transition_model`
- Samples one outcome by probability

This part generates trajectories (what happened in one run).

### 2.3 Policy Evaluation

Implemented in `bellman_policy_evaluation(env, policy, gamma, theta)`.

- Input policy is deterministic mapping: `state -> action`
- Repeats Bellman updates until convergence (`delta < theta`)
- Returns state values `V`

### 2.4 Action-Value Calculation

Implemented in `action_value(env, state, action, V, gamma)`.

This computes:

$$
q(s,a)=\sum_{s',r} p(s',r\mid s,a)\left[r+\gamma V(s')\right]
$$

### 2.5 Policy Update (The New Part)

Implemented in `policy_iteration(env, gamma, theta)`.

For each iteration:

1. evaluate current policy (`bellman_policy_evaluation`)
2. compute `Q(draw)` and `Q(stop)` using `action_value`
3. choose the better action for each state
4. if no state changes action, stop

This is exactly the evaluation-improvement loop.

---

## 3) What Changed from 4+ to 4++

### In 4+

- You evaluated a policy and printed $v_\pi(s)$.
- Useful for understanding value, but the policy itself was not updated.

### In 4++

- You still evaluate values with Bellman updates.
- Then you **use those values to change the policy**.
- Output now shows per-state:
  - $V(s)$
  - $Q(s,\text{draw})$
  - $Q(s,\text{stop})$
  - selected policy action

So yes, your statement is correct:

- Bellman equation itself is evaluation.
- `4++` adds policy update on top of that evaluation.

---

## 4) Interpretation Guide

When reading the output of each policy-iteration step:

1. Check if `Q(draw)` is larger or `Q(stop)` is larger.
2. The policy at that state should follow the larger Q.
3. If policy stops changing across all states, you reached a stable policy.

This process converts raw value estimates into actionable decisions.

---

## 5) Conclusion

`4++` is the point where the code moves from:

- "How good is this policy?" (evaluation)

to

- "How should I change the policy to be better?" (improvement)

That shift is a core RL idea. Bellman evaluation gives the numbers, and policy improvement turns those numbers into better control.
