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

from sklearn.preprocessing import normalize, minmax_scale, MinMaxScaler, StandardScaler

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")
timeoutamount = 0.095

globalMessage = None
timeout = time

state_size = 17
action_size = 8
batch_size = 20
num_of_episodes = 2000
epsilon_decrease_factor = 10

ray_high_tolerance = 3.5

enable_distance_reward = True
enable_ray_reward = True

correct_move = 0
wrong_move = 0

previous_goal_count = 0

class DQNAgent():
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=4000)
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
        #model.add(Dense(24, input_dim=self.state_size))
        #model.add(LeakyReLU())
        model.add(Dense(24, activation='relu'))
        #model.add(Dense(24))
        #model.add(LeakyReLU())
        model.add(Dense(self.action_size))
        #model.add(Softmax())
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
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
            self.model.fit(state, target_f, epochs=1, verbose=0)

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
        if float(ray) <= 0.7:
            isDone = True
            break
        else:
            isDone = False

    return isDone


def get_ray_array():
    rayReward = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    iter = 0
    for reward in rayReward:
        rayReward[iter] = 0.0
        iter += 1
    return rayReward

# 0 : fwd
# 1 : back
# 2 : left
# 3 : right
# 4 : left+fwd
# 5 : left+fwd+fwd
# 6 : fwd+left+left
# 7 : right+fwd
# 8 : right+fwd+fwd
# 9 : fwd+right+right
# 10 : left+back
# 11 : left+back+left
# 12 : left+back+back
# 13 : right+back
# 14 : right+back+right
# 15 : right+back+back
def action_to_ray_conversion(action, ray_index):
    punish = -1.0
    reward = 0.0

    if action == 0:  # fwd
        if ray_index == 0 or ray_index == 5 or ray_index == 8:
            return punish
        else:
            return reward

    elif action == 1:  # back
        if ray_index == 1 or ray_index == 12 or ray_index == 15:
            return punish
        else:
            return reward

    elif action == 2:  # left
        if ray_index == 2 or ray_index == 6 or ray_index == 11:
            return punish
        else:
            return reward

    elif action == 3:  # right
        if ray_index == 3 or ray_index == 9 or ray_index == 14:
            return punish
        else:
            return reward

    elif action == 4:  # fl
        if ray_index == 4 or ray_index == 5 or ray_index == 6:
            return punish
        else:
            return reward

    elif action == 5:  # fr
        if ray_index == 7 or ray_index == 8 or ray_index == 9:
            return punish
        else:
            return reward

    elif action == 6:  # bl
        if ray_index == 10 or ray_index == 11 or ray_index == 12:
            return punish
        else:
            return reward

    elif action == 7:  # br
        if ray_index == 13 or ray_index == 14 or ray_index == 15:
            return punish
        else:
            return reward

    return 0.0


def ns_ray(data, last_rays, action):
    active_rays = 0

    _ray_rewards = get_ray_array()
    _iterator = 0
    for ray in data.rays:
        _reward = 0.0
        if ray <= ray_high_tolerance:
            active_rays += 1
            if last_rays[_iterator] > ray:
                _reward -= int(abs(ray-ray_high_tolerance))
            _reward -= int(action_to_ray_conversion(action, _iterator))

        _ray_rewards[_iterator] = _reward
        _iterator += 1

    if active_rays > 0:
        _iterator = 0
        for _ray_reward in _ray_rewards:
            _ray_rewards[_iterator] -= active_rays
            _iterator += 1

    return _ray_rewards


def ns_distance(dist, orig_dist, last_dist):
        global correct_move, wrong_move

        if last_dist < dist: # punish when agent moves away from target
            wrong_move -= 1
            reward = -0.005
        else: # reward when agent moves closer to target
            correct_move += 1
            reward = 0.001

        if orig_dist < dist:
            reward -= 0.01

        move_factor = 1.0
        if correct_move > 0.0:
            move_factor += correct_move
            wrong_move = 0.0
        elif wrong_move < 0.0:
            move_factor -= wrong_move
            correct_move = 0.0

        return reward*move_factor


def get_reward(message, old_dist, last_rays, action):
    reward = 0
    if message.hasReachedGoal:
        return 1000

    if enable_distance_reward:
        # if message.distanceToGoal < old_dist:
        #     reward = 1.0
        # else:
        #     reward = -2.0

        if message.origDistanceToGoal < message.distanceToGoal:
            reward -= 2.0
        else:
            reward = 1.0

    if enable_ray_reward:
        active_rays = 0
        for ray in message.rays:
            if ray <= ray_high_tolerance:
                active_rays += 1

        if active_rays > 0:
            reward -= active_rays

    return reward


def get_next_state(message, last_dist, last_rays, action):
    dist_state = 0.0

    if enable_ray_reward:
        ray_state = ns_ray(message, last_rays, action)
    else:
        ray_state = get_ray_array()

    if enable_distance_reward:
        dist_state = ns_distance(message.distanceToGoal, message.origDistanceToGoal, last_dist)

    return np.array([[dist_state,
                      ray_state[0],
                      ray_state[1],
                      ray_state[2],
                      ray_state[3],
                      ray_state[4],
                      ray_state[5],
                      ray_state[6],
                      ray_state[7],
                      ray_state[8],
                      ray_state[9],
                      ray_state[10],
                      ray_state[11],
                      ray_state[12],
                      ray_state[13],
                      ray_state[14],
                      ray_state[15]]])


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
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! GOOOOOOOOOOAAAAAAAAALLLLLLLLL !!!!!!!!!!!!!!!!!!!!!!!!!!!! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ")
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
    model = load_model("model_.h5")
    agent.model = model
    return agent


def get_initial_state(dist, rays):
    global init_dist
    return np.array([[
        dist,
        rays,
        rays,
        rays,
        rays,
        rays,
        rays,
        rays,
        rays,
        rays,
        rays,
        rays,
        rays,
        rays,
        rays,
        rays,
        rays
    ]])

agent = DQNAgent(state_size, action_size)
#agent = load_old_model(agent)

while True:
    #  Wait for next request from client
    incoming = handleJsonResponse(socket.recv())
    globalMessage = incoming

    init_dist = globalMessage.origDistanceToGoal
    init_ray = globalMessage.rays[0]

    for iter in range(num_of_episodes):
        state = get_initial_state(init_dist, init_ray)
        state = np.reshape(state, [1, state_size])
        reward = 0.0

        if not previous_goal_count == agent.goal_count and not agent.goal_count == 0 and agent.goal_count % 20 == 0:
            print("save " + str(iter))
            agent.model.save('model_' + str(agent.goal_count) + '_e' + str(iter) + '.h5')
            timeout.sleep(0.2)

        # Learning goes on here
        # We make sure that the agent does not get stuck
        iterator = 0
        for time in range(1000):

            if iterator == 0:
                timeout.sleep(0.2)

            if not globalMessage.done:
                previous_goal_count = agent.goal_count
                action = agent.act(state)
                globalMessage.move = translate_action(action)

                last_distance = globalMessage.distanceToGoal
                last_rays = globalMessage.rays
                globalMessage = update_message(globalMessage)

                next_state = get_next_state(globalMessage, last_distance, last_rays, action)
                next_state = np.reshape(next_state, [1, state_size])

                reward = get_reward(globalMessage, last_distance, last_rays, action)

                done = is_done(globalMessage)

                # Give a penalty if agent is just still
                reward = reward if not done else -100

                print_distance(next_state[0][0])
                #print_rays(next_state[0])
                #print(reward)

                # We want the agent to remember
                agent.mem_remember(state, action, reward, next_state, done)

                next_state = np.reshape(state, [1, state_size])

                state = next_state

                if globalMessage.hasReachedGoal:
                    done = True

                if done:
                    print("Episode: " + str(iter) + " Time: " + str(time) + " Epsilon: " + str(agent.epsilon) + " Goal count: " + str(agent.goal_count) + " Reward: " + str(reward))
                    if agent.goal_count % 2 == 0 and not previous_goal_count == agent.goal_count:
                        globalMessage.addWall = True
                    globalMessage = reset_program(globalMessage)
                    break

            if len(agent.memory) > batch_size:
                agent.train_replay(batch_size, iter)

            iterator += 1

    reset_program(globalMessage)
    print("Total goal count: " + str(agent.goal_count))
    agent.model.save('model_finished.h5')
    timeout.sleep(1)
    socket.context.__exit__()
