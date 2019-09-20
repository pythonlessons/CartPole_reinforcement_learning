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


# Cartpole Game
CartPole is one of the simplest environments in OpenAI gym (collection of environments to develop and test RL algorithms). Cartpole is built on a Markov chain model that is illustrated below.

<p align="center">
    <img src="https://github.com/pythonlessons/CartPole_reinforcement_learning/blob/master/IMAGES/image.png"
</p>
  
Then for each iteration, an agent takes current state (S_t), picks best (based on model prediction) action (A_t) and executes it on an environment. Subsequently, environment returns a reward (R_t+1) for a given action, a new state (S_t+1) and an information if the new state is terminal. The process repeats until termination.

The goal of CartPole is to balance a pole connected with one joint on top of a moving cart. To make it simplier for us, instead of pixel information, there are 4 kinds of information given by the state, such as angle of the pole and position of the cart. An agent can move the cart by performing a series of actions of 0 or 1 to the cart, pushing it left or right.

Gym makes interacting with the game environment really simple:
```
next_state, reward, done, info = env.step(action)
```

Here, ```action``` can be either 0 or 1. If we pass those numbers, env, which represents the game environment, will emit the results. ```done``` is a boolean value telling whether the game ended or not. ```next_state``` space handles all possible state values:<br>
(<br>
[Cart Position from -4.8 to 4.8],<br>
[Cart Velocity from -Inf to Inf],<br>
[Pole Angle from -24° to 24°],<br>
[Pole Velocity At Tip from -Inf to Inf]<br>
)

The old state information paired with ```action```, ```next_state``` and ```reward``` is the information we need for training the agent.

So to understand everything from basics, lets first create CartPole environment where our python script would play with it randomly:

```
import gym
import random

env = gym.make("CartPole-v0")
env.reset()

def Random_games():
    # Each of this episode is its own game.
    for episode in range(10):
        env.reset()
        # this is each frame, up to 500...but we wont make it that far with random.
        for t in range(500):
            # This will display the environment
            # Only display if you really want to see it.
            # Takes much longer to display it.
            env.render()
            
            # This will just create a sample action in any environment.
            # In this environment, the action can be 0 or 1, which is left or right
            action = env.action_space.sample()

            # this executes the environment with an action, 
            # and returns the observation of the environment, 
            # the reward, if the env is over, and other info.
            next_state, reward, done, info = env.step(action)
            
            # lets print everything in one line:
            print(t, next_state, reward, done, info, action)
            if done:
                break
                
Random_games()
```
