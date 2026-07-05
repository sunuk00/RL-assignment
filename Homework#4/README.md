# Reinforcement Learning — HW4

Deep Q-Network, REINFORCE, Actor-Critic을 재고관리(Inventory Control) 환경에 적용한 구현.

---

## 1. Deep Q-Network (DQN)

### 핵심 아이디어

Q-learning을 신경망으로 확장한 value-based method.  
정책을 직접 학습하지 않고, **각 상태-행동 쌍의 가치 Q(s, a)를 추정**한 뒤 greedy하게 행동을 선택한다.

### Bellman Optimality Equation

$$Q^* (s, a) = \mathbb{E}\left[r + \gamma \max_{a'} Q^{*}(s', a') \mid s, a\right]$$
- $Q^{*}(s, a)$: 최적 Q값 (optimal action-value function)
- $r$: 현재 상태에서 행동 $a$를 취했을 때 얻는 보상
- $s'$: 현재 상태에서 행동 $a$를 취한 후 도달하는 다음 상태
- $\gamma$: 할인율 (discount factor), 미래 보상의 중요도를 결정

Q*가 이 방정식을 만족하면 최적 정책이다.

### TD Target (학습 타깃)

$$y = r + \gamma \max_{a'} Q_{\theta^-}(s', a')$$

- $\theta^-$: target network의 파라미터 (일정 주기로 $\theta$로부터 복사)
- target network가 없으면 타깃이 매 스텝 바뀌어 학습이 불안정해진다

### Loss Function

$$\mathcal{L}(\theta) = \mathbb{E}\left[\left(y - Q_\theta(s, a)\right)^2\right]$$


### 행동 선택 (ε-greedy)

$$a = \begin{cases} \text{random action} & \text{with probability } \epsilon \\ \arg\max_a Q_\theta(s, a) & \text{with probability } 1 - \epsilon \end{cases}$$

$\epsilon$은 학습 초반에 크고 점차 감소 → 탐색에서 활용으로 전환.

### 핵심 구성요소

| 구성요소 | 역할 |
|---|---|
| Q-network ($\theta$) | 현재 Q값 추정 |
| Target network ($\theta^-$) | 학습 타깃 안정화 |
| Experience replay buffer | 샘플 간 상관성 제거, 과거 경험 재사용 |

---

## 2. REINFORCE

### 핵심 아이디어

Q값을 추정하는 대신 **policy $\pi_\theta$를 직접 파라미터화**하여 gradient로 업데이트하는 policy gradient method.  
에피소드 전체의 return을 이용한 Monte Carlo 방식으로 업데이트한다.

### Policy Gradient Theorem

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot G_t\right]$$

- $J(\theta)$: 정책 $\pi_\theta$의 기대 수익 (expected return)
- $G_t$: 시간 $t$에서의 return (누적 할인 보상)

### Return (누적 할인 보상)

$$G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$$

- 에피소드가 끝난 뒤에야 $G_t$를 계산할 수 있다
- 편향(bias)은 없지만 분산(variance)이 크다

### REINFORCE Update Rule

$$\theta \leftarrow \theta + \alpha \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot G_t$$

직관: $G_t$가 클수록(좋은 행동) 해당 행동의 확률을 높이고, $G_t$가 작을수록 확률을 낮춘다.

### Baseline을 이용한 분산 감소

baseline $b_t$를 빼도 gradient의 기댓값은 변하지 않는다 (unbiasedness 유지):

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot (G_t - b_t)\right]$$

본 구현에서는 **greedy rollout baseline** 사용:  
현재 정책으로 greedy하게 끝까지 시뮬레이션한 return을 $b_t$로 사용.

$$\delta_t = G_t - b_t \quad \text{(advantage)}$$

- $b_t > G_t$: 현재 행동이 greedy보다 나빴다 → 확률 감소
- $b_t < G_t$: 현재 행동이 greedy보다 좋았다 → 확률 증가

---

## 3. Actor-Critic

### 핵심 아이디어

**Actor**(policy)와 **Critic**(value function)을 동시에 학습.  
REINFORCE의 고분산 문제를 TD error 기반 advantage로 완화하고, 매 스텝마다 업데이트한다.

### 구성

| 네트워크 | 역할 | 출력 |
|---|---|---|
| Actor $\pi_\theta(a \mid s)$ | 행동 선택 (policy) | 행동에 대한 확률 분포 |
| Critic $V_\phi(s)$ | 상태의 가치 추정 | 스칼라 $V(s)$ |

### TD Error (Advantage Estimate)

$$\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$

- $r_t + \gamma V_\phi(s_{t+1})$: TD target (한 스텝 앞을 내다본 추정값)
- $V_\phi(s_t)$: 현재 상태의 가치 추정
- $\delta_t > 0$: 예상보다 좋은 결과 → Actor가 해당 행동 확률을 높임
- $\delta_t < 0$: 예상보다 나쁜 결과 → Actor가 해당 행동 확률을 낮춤

### Actor Update (Policy Gradient)

$$\theta \leftarrow \theta + \alpha_\text{actor} \cdot \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot \delta_t$$

### Critic Update (Value Function)

$$\phi \leftarrow \phi - \alpha_\text{critic} \cdot \nabla_\phi \left(r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)\right)^2$$

즉 Critic은 TD error의 MSE를 최소화하도록 학습된다.

---

## 세 알고리즘 비교

| | DQN | REINFORCE | Actor-Critic |
|---|---|---|---|
| 방식 | Value-based | Policy-based | 둘 다 |
| 업데이트 단위 | 매 스텝 (TD) | 에피소드 끝 (MC) | 매 스텝 (TD) |
| 편향 (Bias) | 있음 | 없음 | 있음 |
| 분산 (Variance) | 낮음 | 높음 | 중간 |
| Exploration | ε-greedy | 확률적 샘플링 | 확률적 샘플링 |
| 추가 구조 | Replay buffer, Target network | Baseline | Actor + Critic 두 네트워크 |