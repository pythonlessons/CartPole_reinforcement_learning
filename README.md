# Solving the CartPole balancing game

The idea of CartPole is that there is a pole standing up on top of a cart. The goal is to balance this pole by moving the cart from side to side to keep the pole balanced upright.

The environment is deemed successful if we can balance for 500 frames, and failure is deemed when the pole is more than 15 degrees from fully vertical or the cart moves more than 2.4 units from the center.

Every frame that we go with the pole "balanced" (less than 15 degrees from vertical), our "score" gets +1, and our target is a score of 500.

Now, how do we do this? There are endless ways, some very complex, and some very specific. I chose to demonstrate how deep reinforcement learning (deep Q-learning) can be implemented and applied to play a CartPole game using Keras and Gym. I will try to explain everything without requiring any prerequisite knowledge about reinforcement learning.

Before starting, take a look at this [YouTube video](https://youtu.be/XiigTGKZfks) with a real-life demonstration of a cartpole problem learning process. Looks amazing, right? Implementing such a self-learning system is easier than you may think. Let’s dive in!

# Reinforcement Learning
In order to achieve the desired behavior of an agent that learns from its mistakes and improves its performance, we need to get more familiar with the concept of <b>Reinforcement Learning (RL)</b>.
RL is a type of machine learning that allows us to create AI agents that learn from the environment by interacting with it in order to maximize its cumulative reward. The same way how we learn to ride a bicycle, AI learns it by trial and error, agents in RL algorithms are incentivized with punishments for bad actions and rewards for good ones.
After each action, the agent receives the feedback. The feedback consists of the reward and next state of the environment. The reward is usually defined by a human. If we use the analogy of the bicycle, we can define reward as the distance from the original starting point.

