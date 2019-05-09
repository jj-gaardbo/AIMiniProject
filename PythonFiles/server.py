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
timeoutamount = 0.07

globalMessage = None
timeout = time

state_size = 17
action_size = 8
batch_size = 64
num_of_episodes = 5000
epsilon_decrease_factor = 20
movement_factor = 0.7

ray_high_tolerance = 2
ray_low_tolerance = 0.9

punish_distance_larger = 5
reward_distance_smaller = 10


class DQNAgent():
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=50000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998
        self.learning_rate = 0.001
        self.model = self.model()
        self.goal_count = 0

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
        if self.epsilon > self.epsilon_min:
            if episode % epsilon_decrease_factor == 0:
                print("!!!!!!!!!!!!!!!!!DECREASE : " + str(episode) + "%" + str(epsilon_decrease_factor))
                self.epsilon *= self.epsilon_decay

    def increment_goal_count(self):
        self.goal_count += 1


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


def translate_action(action):
    if action == 0:
        return 'f'
    elif action == 1:
        return 'b'
    elif action == 2:
        return 'l'
    elif action == 3:
        return 'r'
    elif action == 4:
        return 'fl'
    elif action == 5:
        return 'fr'
    elif action == 6:
        return 'bl'
    elif action == 7:
        return 'br'
    else:
        return 'f'

def is_done(message):
    isDone = False

    for ray in message.rays:
        if float(ray) <= 0.7:
            isDone = True
            break
        else:
            isDone = False

    if message.hasReachedGoal:
        message.hasReachedGoal = False
        isDone = True

    return isDone


def ray_reward(data, movement, returnArray):
    active_rays = 0

    if returnArray:
        rayRewards = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        iterator = 0
        for ray in data.rays:
            if ray-movement <= ray_low_tolerance:
                active_rays += 1
                rayRewards[iterator] -= 10

            elif ray-movement <= ray_high_tolerance:
                rayRewards[iterator] -= 1

            elif ray-movement > ray_high_tolerance:
                rayRewards[iterator] += 0.1

        for reward in rayRewards:
            reward -= active_rays

        return rayRewards

    else:
        reward = 0
        for ray in data.rays:
            if ray <= ray_low_tolerance:
                active_rays += 1
                reward -= 10
            elif ray <= ray_high_tolerance:
                reward -=1
            elif ray > ray_high_tolerance:
                reward += 0.1

            reward -= active_rays
        return reward


def distance_reward(dist, origDist, movement, last_dist): # origDist: 17.189245223999023
        reward = 0
        if last_dist+movement <= dist:
            reward -= punish_distance_larger+(abs(dist-origDist)+dist*1.1)
        elif last_dist+movement > dist:
            reward += reward_distance_smaller+(abs(dist-origDist)-dist)
        return reward


def calculate_reward(message, old_dist):
    reward = 0
    if message.hasReachedGoal:
        agent.increment_goal_count()
        reward += 1000

    reward += distance_reward(message.distanceToGoal, message.origDistanceToGoal, 0, old_dist)
    rayReward = ray_reward(message, 0, False)
    reward += rayReward

    return reward


def get_next_state(distance, origDistance, message, last_dist):
    reward = 0
    reward += distance_reward(distance, origDistance, movement_factor, last_dist)
    rayRewards = ray_reward(message, movement_factor, True)

    return np.array([[reward,
                      rayRewards[0],
                      rayRewards[1],
                      rayRewards[2],
                      rayRewards[3],
                      rayRewards[4],
                      rayRewards[5],
                      rayRewards[6],
                      rayRewards[7],
                      rayRewards[8],
                      rayRewards[9],
                      rayRewards[10],
                      rayRewards[11],
                      rayRewards[12],
                      rayRewards[13],
                      rayRewards[14],
                      rayRewards[15]]])


def update_message(message):
    #  Send reply back to client
    #  In the real world usage, after you finish your work, send your output here
    newMessage = Message(message.move,
                      message.reward,
                      message.hasReachedGoal,
                      message.rays,
                      message.done,
                      message.distanceToGoal,
                      message.origDistanceToGoal)
    socket.send_string(newMessage.to_string())
    incoming = handleJsonResponse(socket.recv())
    if incoming.hasReachedGoal:
        print("GOOOOOOOOOOAAAAAAAAALLLLL!!!!!!!!")
    timeout.sleep(timeoutamount)
    return incoming


def reset_program(message):
    newMessage = Message(message.move,
                         message.reward,
                         message.hasReachedGoal,
                         message.rays,
                         True,
                         message.distanceToGoal,
                         message.origDistanceToGoal)
    socket.send_string(newMessage.to_string())
    globalMessage = handleJsonResponse(socket.recv())
    timeout.sleep(timeoutamount)
    globalMessage.move = 'f'
    globalMessage.reward = 0
    globalMessage.hasReachedGoal = False
    return globalMessage


agent = DQNAgent(state_size, action_size)
while True:

    #  Wait for next request from client
    incoming = handleJsonResponse(socket.recv())
    globalMessage = incoming

    for iter in range(num_of_episodes):
        state = np.array([[globalMessage.distanceToGoal,
                           globalMessage.rays[0],
                           globalMessage.rays[1],
                           globalMessage.rays[2],
                           globalMessage.rays[3],
                           globalMessage.rays[4],
                           globalMessage.rays[5],
                           globalMessage.rays[6],
                           globalMessage.rays[7],
                           globalMessage.rays[8],
                           globalMessage.rays[9],
                           globalMessage.rays[10],
                           globalMessage.rays[11],
                           globalMessage.rays[12],
                           globalMessage.rays[13],
                           globalMessage.rays[14],
                           globalMessage.rays[15]]])
        state = np.reshape(state, [1, state_size])

        # Learning goes on here
        # We make sure that the agent does not get stuck
        for time in range(100):
            if not globalMessage.done:

                action = agent.act(state)
                globalMessage.move = translate_action(action)

                last_distance = globalMessage.distanceToGoal
                globalMessage = update_message(globalMessage)

                next_state = get_next_state(globalMessage.distanceToGoal, globalMessage.origDistanceToGoal, globalMessage, last_distance)
                reward = calculate_reward(globalMessage, last_distance)

                print(reward)

                done = is_done(globalMessage)

                # Give a penalty if agent is just still
                reward = reward if not done else -100

                # We want the agent to remember
                agent.mem_remember(state, action, reward, next_state, done)

                next_state = np.reshape(state, [1, state_size])

                state = next_state

                if done:
                    print("Episode: " + str(iter) + " Time: " + str(time) + " Epsilon: " + str(agent.epsilon) + " Goal count: " + str(agent.goal_count))
                    globalMessage = reset_program(globalMessage)
                    break

                if len(agent.memory) > batch_size:
                    agent.train_replay(batch_size, iter)

    reset_program(globalMessage)
    print("Total goal count: " + str(agent.goal_count))
    agent.model.save('robot_model.h5')
    timeout.sleep(1)
    socket.context.__exit__()
