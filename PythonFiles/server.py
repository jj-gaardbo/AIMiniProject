import random
import time
import zmq
import json
import numpy as np
from collections import deque
from keras.models import Sequential, load_model
from keras.layers import Dense, LeakyReLU, Softmax
from keras.optimizers import Adam
from keras.utils import plot_model
import csv

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")
timeoutamount = 0.1

globalMessage = None
timeout = time

state_size = 17
action_size = 8
batch_size = 32
num_of_episodes = 2000
epsilon_decrease_factor = 10

previous_goal_count = 0
leaky = False


def append_to_csv(episode, data):
    with open(r'monitor.csv', mode='a') as reward_monitor:
        reward_monitor = csv.writer(reward_monitor, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        reward_monitor.writerow([episode+1, np.mean(data)])


class DQNAgent():
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998
        self.learning_rate = 0.001
        self.model = self.model()
        self.goal_count = 0
        self.history = None

    def model(self):
        model = Sequential()
        if leaky:
            model.add(Dense(24, input_dim=self.state_size))
            model.add(LeakyReLU())
            model.add(Dense(24))
            model.add(LeakyReLU())
        else:
            model.add(Dense(24, input_dim=self.state_size, activation='relu'))
            model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size))
        #model.add(Softmax())
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate), metrics=["accuracy"])
        plot_model(model, to_file='model.png', show_shapes=True)
        return model

    # Make the agent remember stuff
    def mem_remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        #print("length" + str(self.memory.__len__()))

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
            self.history = self.model.fit(state, target_f, epochs=1, verbose=0)

        # We have to decrease the epsilon each time to make the actions less and less random
        if self.epsilon > self.epsilon_min:
            if episode % epsilon_decrease_factor == 0 and not episode == 0:
                #print("!!!!!!!!!!!!!!!!!DECREASE : " + str(episode) + "%" + str(epsilon_decrease_factor))
                self.epsilon *= self.epsilon_decay

    def increment_goal_count(self):
        self.goal_count += 1


# JSON Message Class to send to Unity
class Message():
    def __init__(self, move, reward, hasReachedGoal, rays, done, distanceToGoal, origDistanceToGoal, addWall, goalCount):
        self.move = move
        self.reward = reward
        self.hasReachedGoal = hasReachedGoal
        self.rays = rays
        self.done = done
        self.distanceToGoal = distanceToGoal
        self.origDistanceToGoal = origDistanceToGoal
        self.addWall = addWall
        self.goalCount = goalCount

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
                "addWall": self.addWall,
                "goalCount": self.goalCount
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
        print("goalCount", self.goalCount)


def handleJsonResponse(data):
    dataReceived = json.loads(data.decode('utf-8'))
    type(dataReceived)
    return Message(dataReceived['move'], dataReceived['reward'], dataReceived['hasReachedGoal'], dataReceived['rays'], dataReceived['done'], dataReceived['distanceToGoal'], dataReceived['originalDistanceToGoal'], dataReceived['addWall'], dataReceived['goalCount'])


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
        if float(ray) <= 0.65:
            isDone = True
            break
        else:
            isDone = False

    return isDone


def get_reward(message, last_dist):
    if message.hasReachedGoal:
        return 100

    _reward = 2 ** (message.origDistanceToGoal/message.distanceToGoal)

    if message.distanceToGoal > message.origDistanceToGoal:
        _reward = 2 ** (message.distanceToGoal/message.origDistanceToGoal)*-1

    # if last_dist < message.distanceToGoal:
    #     _reward -= 0.001

    return _reward


def get_next_state(message):
    return np.array([[message.distanceToGoal,
                      message.rays[0],
                      message.rays[1],
                      message.rays[2],
                      message.rays[3],
                      message.rays[4],
                      message.rays[5],
                      message.rays[6],
                      message.rays[7],
                      message.rays[8],
                      message.rays[9],
                      message.rays[10],
                      message.rays[11],
                      message.rays[12],
                      message.rays[13],
                      message.rays[14],
                      message.rays[15]
                      ]])


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
                      message.addWall,
                      agent.goal_count)
    socket.send_string(newMessage.to_string())
    incoming = handleJsonResponse(socket.recv())
    if incoming.hasReachedGoal:
        agent.increment_goal_count()
        incoming.goalCount = agent.goal_count

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
                         message.addWall,
                         agent.goal_count)
    socket.send_string(newMessage.to_string())
    globalMessage = handleJsonResponse(socket.recv())
    timeout.sleep(timeoutamount)
    globalMessage.move = None
    globalMessage.hasReachedGoal = False
    globalMessage.goalCount = agent.goal_count
    return globalMessage


def print_distance(distance):
    print(distance)


def print_rays(rays):
    print("Ray0: {}".format(rays[1]))
    print("Ray1: {}".format(rays[2]))
    print("Ray2: {}".format(rays[3]))
    print("Ray3: {}".format(rays[4]))
    print("Ray4: {}".format(rays[5]))
    print("Ray5: {}".format(rays[6]))
    print("Ray6: {}".format(rays[7]))
    print("Ray7: {}".format(rays[8]))
    print("Ray8: {}".format(rays[9]))
    print("Ray9: {}".format(rays[10]))
    print("Ray10: {}".format(rays[11]))
    print("Ray11: {}".format(rays[12]))
    print("Ray12: {}".format(rays[13]))
    print("Ray13: {}".format(rays[14]))
    print("Ray14: {}".format(rays[15]))
    print("Ray15: {}".format(rays[16]))


def load_old_model(agent):
    model = load_model("model_finished_34.h5")
    agent.model = model
    agent.epsilon = agent.epsilon_min
    return agent


def get_initial_state(dist, rays):
    global init_dist
    return np.array([[
        dist,
        rays[0],
        rays[1],
        rays[2],
        rays[3],
        rays[4],
        rays[5],
        rays[6],
        rays[7],
        rays[8],
        rays[9],
        rays[10],
        rays[11],
        rays[12],
        rays[13],
        rays[14],
        rays[15]
    ]])


agent = DQNAgent(state_size, action_size)
#agent = load_old_model(agent)

while True:
    #  Wait for next request from client
    incoming = handleJsonResponse(socket.recv())
    globalMessage = incoming

    init_dist = globalMessage.origDistanceToGoal
    init_rays = globalMessage.rays

    for iter in range(num_of_episodes):
        state = get_initial_state(init_dist, init_rays)
        state = np.reshape(state, [1, state_size])
        reward = 0.0
        episode_reward = [reward]

        if not previous_goal_count == agent.goal_count and not agent.goal_count == 0 and agent.goal_count % 20 == 0:
            print("save " + str(iter))
            agent.model.save('empty_' + str(agent.goal_count) + '_e' + str(iter) + '.h5')
            timeout.sleep(0.2)

        # Learning goes on here
        # We make sure that the agent does not get stuck
        iterator = 0
        for time in range(500):

            if iterator == 0:
                timeout.sleep(0.2)

            if not globalMessage.done:
                previous_goal_count = agent.goal_count
                action = agent.act(state)
                globalMessage.move = translate_action(action)

                last_distance = globalMessage.distanceToGoal
                last_rays = globalMessage.rays
                globalMessage = update_message(globalMessage)

                next_state = get_next_state(globalMessage)

                reward = get_reward(globalMessage, last_distance)

                done = is_done(globalMessage)

                # Give a penalty if agent is just still
                reward = reward if not done else -100

                #print_distance(next_state[0][0])
                #print(globalMessage.rays)
                #print_rays(next_state[0])
                #print(next_state[0])
                #print(reward)

                # We want the agent to remember
                agent.mem_remember(state, action, reward, next_state, done)

                episode_reward.append(reward)

                next_state = np.reshape(state, [1, state_size])

                state = next_state

                if globalMessage.hasReachedGoal:
                    done = True

                if done:
                    append_to_csv(iter, episode_reward)
                    print("Episode: " + str(iter) + " Time: " + str(time) + " Epsilon: " + str(agent.epsilon) + " Goal count: " + str(agent.goal_count) + " Reward: " + str(reward))
                    # if agent.goal_count % 2 == 0 and not previous_goal_count == agent.goal_count:
                    #     globalMessage.addWall = True
                    globalMessage = reset_program(globalMessage)
                    break

            if len(agent.memory) > batch_size:
                agent.train_replay(batch_size, iter)

            iterator += 1

    reset_program(globalMessage)
    print("Total goal count: " + str(agent.goal_count))
    agent.model.save('model_finished_34.h5')
    timeout.sleep(1)
    socket.context.__exit__()
