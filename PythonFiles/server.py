import random
import time
import zmq
import json
import numpy as np
from collections import deque
from keras.models import Sequential, load_model
from keras.layers import Dense, LeakyReLU, Softmax
from keras.optimizers import Adam
from sklearn.preprocessing import normalize, minmax_scale


context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")
timeoutamount = 0.03

globalMessage = None
timeout = time

state_size = 17
action_size = 8
batch_size = 64
num_of_episodes = 5000
epsilon_decrease_factor = 5

ray_high_tolerance = 2.5
ray_reward_factor = 0.1

distanceReward = True
rayReward = True

correct_move = 0.0


class DQNAgent():
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998
        self.learning_rate = 0.001
        self.model = self.model()
        self.goal_count = 0

    def model(self):
        model = Sequential()
        #model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, input_dim=self.state_size))
        model.add(LeakyReLU())
        #model.add(Dense(24, activation='relu'))
        model.add(Dense(24))
        model.add(LeakyReLU())
        model.add(Dense(self.action_size))
        #model.add(Softmax())
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
            if episode % epsilon_decrease_factor == 0 and not episode == 0:
                print("!!!!!!!!!!!!!!!!!DECREASE : " + str(episode) + "%" + str(epsilon_decrease_factor))
                self.epsilon *= self.epsilon_decay

    def increment_goal_count(self):
        self.goal_count += 1


# JSON Message Class to send to Unity
class Message():
    def __init__(self, move, reward, hasReachedGoal, rays, done, distanceToGoal, origDistanceToGoal, addWall):
        self.move = move
        self.reward = reward
        self.hasReachedGoal = hasReachedGoal
        self.rays = rays
        self.done = done
        self.distanceToGoal = distanceToGoal
        self.origDistanceToGoal = origDistanceToGoal
        self.addWall = addWall

    def to_string(self):
        return json.dumps(
            {
                "move": self.move,
                "reward": self.reward,
                "hasReachedGoal": self.hasReachedGoal,
                "rays": self.rays,
                "done": self.done,
                "distanceToGoal": self.distanceToGoal,
                "origDistanceToGoal": self.origDistanceToGoal,
                "addWall": self.addWall
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
        print("addWall", self.addWall)


def handleJsonResponse(data):
    dataReceived = json.loads(data.decode('utf-8'))
    type(dataReceived)
    return Message(dataReceived['move'], dataReceived['reward'], dataReceived['hasReachedGoal'], dataReceived['rays'], dataReceived['done'], dataReceived['distanceToGoal'], dataReceived['originalDistanceToGoal'], dataReceived['addWall'])


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


def ray_reward(data, returnArray):
    active_rays = 0

    if returnArray:
        rayRewards = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        iterator = 0
        for ray in data.rays:
            _reward = 0.0
            if ray <= ray_high_tolerance:
                active_rays += 1
                _reward -= abs(ray-ray_high_tolerance)*ray_reward_factor

            rayRewards[iterator] = _reward
            iterator += 1

        if active_rays != 0:
            _iterator = 0
            for _rayReward in rayRewards:
                rayRewards[_iterator] *= active_rays*ray_reward_factor
                _iterator += 1

        return rayRewards

    else:
        reward = 0

        for ray in data.rays:
            if ray <= ray_high_tolerance:
                active_rays += 1
                reward -= abs(ray-ray_high_tolerance)*ray_reward_factor

        if active_rays != 0:
            reward *= active_rays*ray_reward_factor

        return reward


def distance_reward(dist, origDist, last_dist): # origDist: 17.189245223999023
        reward = 0
        global correct_move

        if last_dist < dist: # punish when agent moves away from target
            correct_move = 0.0
            reward -= abs(origDist - dist) * dist
        else: # reward when agent moves closer to target
            correct_move += 1.0
            reward += abs(origDist - dist)*0.998

        reward = reward+correct_move

        return reward


def calculate_reward(message, old_dist):
    reward = 0
    if message.hasReachedGoal:
        agent.increment_goal_count()
        reward += 100

    if rayReward:
        reward += ray_reward(message, False)

    if distanceReward:
        reward += distance_reward(message.distanceToGoal, message.origDistanceToGoal, old_dist)

    return reward


def get_next_state(distance, origDistance, message, last_dist):
    reward = 0

    if rayReward:
        rayRewards = ray_reward(message, True)
    else:
        rayRewards = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    if distanceReward:
        reward += distance_reward(distance, origDistance, last_dist)

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
                      message.origDistanceToGoal,
                      message.addWall)
    socket.send_string(newMessage.to_string())
    incoming = handleJsonResponse(socket.recv())
    if incoming.hasReachedGoal:
        print("GOOOOOOOOOOAAAAAAAAALLLLL!!!!!!!!")

    timeout.sleep(timeoutamount)
    return incoming


def reset_program(message):
    newMessage = Message(None,
                         message.reward,
                         message.hasReachedGoal,
                         message.rays,
                         True,
                         message.distanceToGoal,
                         message.origDistanceToGoal,
                         message.addWall)
    socket.send_string(newMessage.to_string())
    globalMessage = handleJsonResponse(socket.recv())
    timeout.sleep(timeoutamount)
    globalMessage.move = None
    globalMessage.hasReachedGoal = False
    return globalMessage


def print_rays(rays):
    print("Ray1: {}".format(rays[1]))
    print("Ray2: {}".format(rays[2]))
    print("Ray3: {}".format(rays[3]))
    print("Ray4: {}".format(rays[4]))
    print("Ray5: {}".format(rays[5]))
    print("Ray6: {}".format(rays[6]))
    print("Ray7: {}".format(rays[7]))
    print("Ray8: {}".format(rays[8]))
    print("Ray9: {}".format(rays[9]))
    print("Ray10: {}".format(rays[10]))
    print("Ray11: {}".format(rays[11]))
    print("Ray12: {}".format(rays[12]))
    print("Ray13: {}".format(rays[13]))
    print("Ray14: {}".format(rays[14]))
    print("Ray15: {}".format(rays[15]))
    print("Ray16: {}".format(rays[16]))

def load_old_model(agent):
    model = load_model("robot_model_over 10_goals.h5")
    agent.model = model
    agent.epsilon = agent.epsilon_min


agent = DQNAgent(state_size, action_size)
#load_old_model(agent)
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
        for time in range(500):

            if not globalMessage.done:
                previous_goal_count = agent.goal_count
                action = agent.act(state)
                globalMessage.move = translate_action(action)

                last_distance = globalMessage.distanceToGoal
                globalMessage = update_message(globalMessage)

                next_state = get_next_state(globalMessage.distanceToGoal, globalMessage.origDistanceToGoal, globalMessage, last_distance)
                reward = calculate_reward(globalMessage, last_distance)

                #print_rays(next_state[0])

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
                    if agent.goal_count % 3 == 0 and not previous_goal_count == agent.goal_count:
                        globalMessage.addWall = True
                    globalMessage = reset_program(globalMessage)
                    break

                if len(agent.memory) > batch_size:
                    agent.train_replay(batch_size, iter)

    reset_program(globalMessage)
    print("Total goal count: " + str(agent.goal_count))
    if agent.epsilon <= agent.epsilon_min and agent.goal_count > 50:
        print("This agent has figured it out!")
        agent.model.save('robot_model_over 10_goals.h5')
        timeout.sleep(1)
    agent.model.save('robot_model.h5')
    timeout.sleep(1)
    socket.context.__exit__()
