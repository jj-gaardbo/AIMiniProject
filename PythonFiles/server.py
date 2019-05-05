import random
import time
import zmq
import json
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

class DQNAgent():
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998
        self.learning_rate = 0.001
        self.model = self.model()

    def model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # Make the agent remember stuff
    def mem_remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Force the agent to do random exploration which gradually decreases its epsilon
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def train_replay(self, batch_size, episode):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # Here is the actual QLearning algorithm
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))

            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        # We have to decrease the epsilon each time to make the actions less and less random
        if self.epsilon > self.epsilon_min and episode % 10 == 0:
            self.epsilon *= self.epsilon_decay


# JSON Message Class to send to Unity
class Message():
    def __init__(self, move, reward, hasReachedGoal, rays, done, distanceToGoal, origDistanceToGoal):
        self.move = move
        self.reward = reward
        self.hasReachedGoal = hasReachedGoal
        self.rays = rays
        self.done = done
        self.distanceToGoal = distanceToGoal
        self.origDistanceToGoal = origDistanceToGoal

    def to_string(self):
        return json.dumps(
            {
                "move": self.move,
                "reward": self.reward,
                "hasReachedGoal": self.hasReachedGoal,
                "rays": self.rays,
                "done": self.done,
                "distanceToGoal": self.distanceToGoal,
                "origDistanceToGoal": self.origDistanceToGoal
            }
        )

    def print(self):
        print("move", self.move)
        print("reward", self.reward)
        print("hasReachedGoal", self.hasReachedGoal)
        print("rays", self.rays)
        print("done", self.done)
        print("distanceToGoal", self.distanceToGoal)
        print("origDistanceToGoal", self.origDistanceToGoal)


def handleJsonResponse(data):
    dataReceived = json.loads(data.decode('utf-8'))
    type(dataReceived)
    return Message(dataReceived['move'], dataReceived['reward'], dataReceived['hasReachedGoal'], dataReceived['rays'], dataReceived['done'], dataReceived['distanceToGoal'], dataReceived['originalDistanceToGoal'])


def handleRays(data):
    reward = 0
    leftfar = data.rays[0]
    left = data.rays[1]
    front = data.rays[2]
    right = data.rays[3]
    rightfar = data.rays[4]

    for ray in data.rays:
        if ray < 0.5:
            reward -= 0.2

    if leftfar < 1 and left < 1:
        data.move = "r"
        reward -= 0.5
    elif leftfar < 1:
        data.move = "r"
        reward -= 0.01
    elif left < 1:
        data.move = "r"
        reward -= 0.01
    elif front < 1:
        data.move = "b"
        reward -= 0.01
    elif right < 1 and rightfar < 1:
        data.move = "l"
        reward -= 0.5
    elif rightfar < 1:
        data.move = "l"
        reward -= 0.01
    elif right < 1:
        data.move = "l"
        reward -= 0.01
    else:
        data.move = "f"

    return data, reward


def calculate_reward(message):
    reward = 0

    if message.hasReachedGoal:
        reward += 100

    if message.distanceToGoal > message.origDistanceToGoal:
        reward -= 0.1

    if message.distanceToGoal > message.origDistanceToGoal:
        reward += 0.1

    message, rayReward = handleRays(message)

    reward += rayReward

    return message, reward


#calculate state and next state. Look at openAI documentation

#medium.com
#https://towardsdatascience.com/

#(curriculum learning)

state_size = 4
action_size = 4
agent = DQNAgent(state_size, action_size)
while True:

    #  Wait for next request from client
    incoming = handleJsonResponse(socket.recv())

    done = False
    batch_size = 32
    for iter in range(500):
        state = 0
        #state = np.reshape(state, [1, state_size])

        # Learning goes on here
        # We make sure that the agent does not get stuck
        for time in range(500):

            action = agent.act(state)
            next_state = 0
            incoming, reward = calculate_reward(incoming)
            done = incoming.done

            # Give a penalty if agent is just still
            reward = reward if not done else -10

            #next_state = np.reshape(next_state, [1, state_size])

            # We want the agent to remember
            agent.mem_remember(state, action, reward, next_state, done)

            state = next_state

            if done:
                print("Episode: " + str(iter) + " Time: " + str(time) + " Epsilon: " + str(agent.epsilon))
                break

            #if len(agent.memory) > batch_size: #agent.train_replay(batch_size, iter)

    #  Send reply back to client
    #  In the real world usage, after you finish your work, send your output here
    message = Message(incoming.move, 1, incoming.hasReachedGoal, incoming.rays, incoming.done, incoming.distanceToGoal, incoming.origDistanceToGoal)
    socket.send_string(message.to_string())



