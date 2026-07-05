# Deep Q-Networks (DQN)
DQN is a reinforcement learning algorithm that combines Q-learning with deep neural networks. It uses a neural network to approximate the Q-value function, allowing it to handle high-dimensional state spaces. DQN also incorporates techniques such as experience replay and target networks to stabilize training.

There are two main components in DQN:
1. **Q-Network**: A neural network that takes the state as input and outputs the Q-values for each possible action. The network is trained to minimize the difference between the predicted Q-values and the target Q-values.
2. **Target Network**: A separate neural network that is used to compute the target Q
-values. The target network is updated periodically with the weights of the Q-network to provide a stable target for training.


# REINFORCEMENT LEARNING

# Actor-Critic