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

# Learn with Simple Neural Network using Keras
This tutorial is not about deep learning or neural networks. So I will not explain how it works in details, I'll consider it just as a black box algorithm that approximately maps inputs to outputs. This is basically an NN algorithm that learns on the pairs of examples input and output data, detects some kind of patterns, and predicts the output based on an unseen input data.

Neural networks are not the focus of this tutorial, but we should understand how it is used to learn in deep Q-learning algorithm.

Keras makes it really simple to implement a basic neural network. With code below we will create an empty NN model. activation, loss and optimizer are the parameters that define the characteristics of the neural network, but we are not going to discuss it here.

```
from keras.models import  Model
from keras.layers import Input, Dense, Dropout
from keras.optimizers import Adam

# Neural Network model for Deep Q Learning
def OurModel(input_shape, action_space):
    X_input = Input(input_shape)
    X = X_input

    # 'Dense' is the basic form of a neural network layer
    # Input Layer of state size(4) and Hidden Layer with 512 nodes
    X = Dense(512, input_shape=input_shape, activation="relu")(X)
    X = Dropout(0.5)(X)
    
    # Hidden layer with 256 nodes
    X = Dense(256, activation="relu")(X)
    X = Dropout(0.5)(X)
    
    # Hidden layer with 64 nodes
    X = Dense(64, activation="relu")(X)
    X = Dropout(0.5)(X)
    
    # Output Layer with # of actions: 2 nodes (left, right)
    X = Dense(action_space, activation="linear")(X)

    model = Model(inputs = X_input, outputs = X, name='CartPole model')
    model.compile(loss='mse', optimizer=Adam())
    
    return model
```
For a NN to understand and predict based on the environment data, we have initialized our model (will show it in original code) and feed it the information. Later in full code you will see, that fit() method feeds input and output pairs to the model. Then the model will train on those data to approximate the output based on the input.

In above model, I used 3 layers neural network, 512, 256 and 64 neurons. With every layer I added dropout layer, later when we will be training our model, you will see that when training DQN it performs worse than in test mode, this is because of dropout layer. But our goal is to make perfect model on test mode, so everything is fine! Feel free to play with its structure and parameters.

Later in training process you will see what makes the NN to predict the reward value from a certain state. You will see that in code I will use ```model.fit(next_state, reward)```, same as in standard Keras NN model.

After training, the model we will be able to predict the output from unseen input. When we call ```predict()``` function on the model, the model will predict the reward of current state based on the data we trained. Like so: ```prediction = model.predict(next_state)```


# Implementing Deep Q Network (DQN)
Normally in games, the reward directly relates to the score of the game. But, imagine a situation where the pole from CartPole game is tilted to the left. The expected future reward of pushing left button will then be higher than that of pushing the right button since it could yield higher score of the game as the pole survives longer.

In order to logically represent this intuition and train it, we need to express this as a formula that we can optimize on. The loss is just a value that indicates how far our prediction is from the actual target. For example, the prediction of the model could indicate that it sees more value in pushing the left button when in fact it can gain more reward by pushing the right button. We want to decrease this gap between the prediction and the target (loss). So, we will define our loss function as follows:
<p align="center">
    <img src="https://github.com/pythonlessons/CartPole_reinforcement_learning/blob/master/IMAGES/math.PNG"
</p>
We first carry out an action a and observe the reward r and resulting new state s. Based on the result, we calculate the maximum target Q and then discount it so that the future reward is worth less than immediate reward. Lastly, we add the current reward to the discounted future reward to get the target value. Subtracting our current prediction from the target gives the loss. Squaring this value allows us to punish the large loss value more and treat the negative values same as the positive values.

But it's not that difficult than you think it is, Keras takes care of most of the difficult tasks for us. We just need to define our target. We can express the target in a magical one line of code in python: 
```target = reward + gamma * np.max(model.predict(next_state))```

Keras does all the work of subtracting the target from NN output and squaring it. It also applies the learning rate that we can define when creating the neural network model (otherwise model will define it by itself). This all happens inside the fit() function. This function decreases the gap between our prediction to target by the learning rate. The approximation of the Q-value converges to the true Q-value as we repeat the updating process. The loss will decrease, and score will grow higher.

The most notable features of the DQN algorithm are remember and replay methods. Both are simple concepts. The original DQN architecture contains a several more tweaks for better training, but we are going to stick to a simpler version for better understanding.


# Implementing Remember function
One of the specific things for DQN is that neural network used in the algorithm tends to forget the previous experiences as it overwrites them with new experiences. Experience replay is a biologically inspired process that uniformly (to reduce correlation between subsequent actions) samples experiences from the memory and for each entry updates its Q value. So, we need a memory (list) of previous experiences and observations to re-train the model with the previous experiences. We will call this array of experiences memory and use remember() function to append state, action, reward, and next state to the memory.

In our example, the memory list will have a form of:
```memory = [(state, action, reward, next_state, done)...]```

And remember function will simply store states, actions and resulting rewards to the memory like:
```
def remember(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))
```

done is just a Boolean that indicates if the state is the final state (cartpole failed).
