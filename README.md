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
```
memory = [(state, action, reward, next_state, done)...]
```

And remember function will simply store states, actions and resulting rewards to the memory like:
```
def remember(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))
```

done is just a Boolean that indicates if the state is the final state (cartpole failed).


# Implementing Replay function
A method that trains NN with experiences in the memory we will call ```replay()``` function. First, we will sample some experiences from the memory and call them minibath. ```minibatch = random.sample(memory, min(len(memory), batch_size))```

The above code will make a minibatch, which is just a randomly sampled elements from full memories of size ```batch_size```. I will set the batch size as 64 for this example. If memory size is less than 64, we will take everything is in our memory.

To make the agent perform well in long-term, we need to consider not only the immediate rewards but also the future rewards we are going to get. In order to do this, we are going to have a ```discount rate``` or ```gamma``` and ultimately adding it to the current state reward. This way the agent will learn to maximize the discounted future reward based on the given state. In other words, we are updating our Q value with the cumulative discounted future rewards.

For those of you who wonder how such function can possibly converge, as it looks like it is trying to predict its own output (in some sense it is), don’t worry - it’s possible and in our simple case it does. However, convergence is not always that 'easy' and in more complex problems there comes a need of more advanced techniques than CartPole stabilize training. These techniques are for example Double DQN’s or Dueling DQN’s, but that’s a topic for another article (stay tuned).

```
def replay(self):
    x_batch, y_batch = [], []
    # Randomly sample minibatch from the memory
    minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
    # Extract informations from each memory
    for state, action, reward, next_state, done in minibatch:
        # make the agent to approximately map the current state to future discounted reward
        # We'll call that y_target
        y_target = self.model.predict(state)
        # if done, make our target reward
        if done:
            y_target[0][action] = reward
        else:
            # predict the future discounted reward
            y_target[0][action] = reward + self.gamma * np.max(self.model.predict(next_state)[0])
        # append results to lists, that will be used for training
        x_batch.append(state[0])
        y_batch.append(y_target[0])
        
    # Train the Neural Network with batches
    self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
    if self.epsilon > self.epsilon_min:
        self.epsilon *= self.epsilon_decay
```

# Setting Hyper Parameters
There are some parameters that have to be passed to a reinforcement learning agent. You will see similar parameters in all DQN models:

* EPISODES - number of games we want the agent to play.
* gamma - decay or discount rate, to calculate the future discounted reward.
* epsilon - exploration rate, this is the rate in which an agent randomly decides its action rather than prediction.
* epsilon_decay - we want to decrease the number of explorations as it gets good at playing games.
* epsilon_min - we want the agent to explore at least this amount.
* learning_rate - Determines how much neural net learns in each iteration (if used).
* batch_size - Determines how much memory DQN will use to learn.

# Putting It All Together: Coding The Deep Q-Learning Agent
I tried to explain each part of the agent in the above. In the code below I'll implement everything we’ve talked about as a nice and clean class called DQNAgent.

```
import random
import gym
import numpy as np
from collections import deque
from keras.models import Model, load_model
from keras.layers import Input, Dense, LSTM, Reshape, Dropout
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

class DQNAgent:
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.EPISODES = 1000
        self.memory = deque(maxlen=2000)
        
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.0001
        self.epsilon_decay = 0.999
        self.batch_size = 128

        self.model = OurModel(input_shape=(self.state_size,), action_space = self.action_size)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state))

    def replay(self):
        x_batch, y_batch = [], []
        # Randomly sample minibatch from the memory
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        for state, action, reward, next_state, done in minibatch:
            # make the agent to approximately map the current state to future discounted reward
            # We'll call that y_target
            y_target = self.model.predict(state)
            # if done, make our target reward
            if done:
                y_target[0][action] = reward
            else:
                # predict the future discounted reward
                y_target[0][action] = reward + self.gamma * np.max(self.model.predict(next_state)[0])
            # append results to lists, that will be used for training
            x_batch.append(state[0])
            y_batch.append(y_target[0])

        # Train the Neural Network with batches
        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def load(self, name):
        self.model = load_model(name)

    def save(self, name):
        self.model.save(name)

    def run(self):
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                self.env.render()
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                if not done:
                    reward = reward
                else:
                    reward = -10
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
                if done:
                    print("episode: {}/{}, score: {}, e: {:.2}".format(e, self.EPISODES, i, self.epsilon))
                    if i == 500:
                        print("Saving trained model as cartpole-dqn.h5")
                        self.save("cartpole-dqn.h5")
                    break
                self.replay()

    def test(self):
        self.load("cartpole-dqn.h5")
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                self.env.render()
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, _ = self.env.step(action)
                state = np.reshape(next_state, [1, self.state_size])
                i += 1
                if done:
                    print("episode: {}/{}, score: {}".format(e, self.EPISODES, i))
                    break

if __name__ == "__main__":
    agent = DQNAgent()
    agent.run()
    #agent.test()
```

# DQN CartPole training part

Below is part code, responsible for training our DQN model. I will not go deep into explanation line by line, because everything was explained above. But in our code, we are running for 1000 episodes of game to train. If you don't want to see how training performs you can comment this line self.env.render(). Every step is rendered here, and while done is equal to False, our model keeps training. We save results from every step to memory, which we use for training on every step. When our model hits score of 500, we save it and already we can use it for testing. But I recommend not to turn off training at first save, give it more time to train before testing. It may take up to 100 steps before it reaches 500 score. You may ask, why it takes so long? Answer is simple, because of Dropout layer in our model, without dropout it may reach 500 much faster, but then our testing results would be worse. So, here is the code part of this short explanation:
```
def run(self):
    for e in range(self.EPISODES):
        state = self.env.reset()
        state = np.reshape(state, [1, self.state_size])
        done = False
        i = 0
        while not done:
            self.env.render()
            action = self.act(state)
            next_state, reward, done, _ = self.env.step(action)
            next_state = np.reshape(next_state, [1, self.state_size])
            if not done:
                reward = reward
            else:
                reward = -10
            self.remember(state, action, reward, next_state, done)
            state = next_state
            i += 1
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}".format(e, self.EPISODES, i, self.epsilon))
                if i == 500:
                    print("Saving trained model as cartpole-dqn.h5")
                    self.save("cartpole-dqn.h5")
                break
            self.replay()
```

For me, model reached 500 score in 73rd step, here my model was saved:
<p align="center">
    <img src="https://github.com/pythonlessons/CartPole_reinforcement_learning/blob/master/IMAGES/training_model.PNG"
</p>


# DQN CartPole testing part
So now, when you have trained your model, its time test it! Comment ```agent.run()``` line and uncomment ```agent.test()```. And check, how your first DQN model works!
```
if __name__ == "__main__":
    agent = DQNAgent()
    #agent.run()
    agent.test()
```

So here is 20 test episodes of our trained model, as you can see 16 times it hit the maximum score, it would be interesting what is the maximum score it could hit, but sadly limit is 500
<p align="center">
    <img src="https://github.com/pythonlessons/CartPole_reinforcement_learning/blob/master/IMAGES/testing_model.PNG"
</p>

And here is short gif, which shows how our agent performs:
<p align="center">
    <img src="https://github.com/pythonlessons/CartPole_reinforcement_learning/blob/master/IMAGES/CartPole_test.gif"
</p>
    
For this task our goal was reached, short recap what we done:

* Learned how DQN works
* Wrote simple DQN model
* Teched model to play CartPole game

This is the end for this tutorial. I challenge you to try creating your own RL agents! Let me know how they perform in solving the cartpole problem. Furthermore, stay tuned for more future tutorials.
