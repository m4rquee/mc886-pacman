import math
from random import choice

from game import Agent, Actions, Directions
from util import Counter, flipCoin, nearestPoint, manhattanDistance

class RlearningAgent(Agent):

    def __init__(self, alpha=0.1, epsilon=0.9, gamma=0.9, numTraining = 100):
        # alpha       - learning rate
        # epsilon     - exploration rate
        # gamma       - discount factor
        # numTraining - number of training episodes
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.numTraining = int(numTraining)

        self.episodesSoFar = 0
        # Q-values
        self.q_value = Counter()
        # current score
        self.score = 0
        # last states
        self.lastStates = []
        # last actions
        self.lastActions = []

    # get Q(s,a)
    def getQValue(self, state, action):
        return self.q_value[(state, action)]

    # return the maximum Q of state
    def getMaxQ(self, state, actions):
        q_list = []
        for a in actions:
            q = self.getQValue(state,a)
            q_list.append(q)
        if len(q_list) ==0:
            return 0
        return max(q_list)

    # update Q value
    def updateQ(self, state, action, reward, qmax):
        q = self.getQValue(state, action)
        self.q_value[(state, action)] = q + self.alpha * (reward + self.gamma * qmax - q)

    # return the action that maximises Q for given state
    def greedyAction(self, state, actions):
        tmp = Counter()
        for action in actions:
          tmp[action] = self.getQValue(state, action)
        return tmp.argMax()

    def getAction(self, state):
        # The data we have about the state of the game
        # the legal action of this state
        legal = state.getLegalPacmanActions()

        # Avoid stopping
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        pacman_position = state.getPacmanPosition()
        ghost_positions = state.getGhostPositions()
        food_locations = state.getFood()
        capsule_locations = state.getCapsules()

        curr_state = str(legal) + str(pacman_position) + str(ghost_positions) + str(food_locations) + str(capsule_locations)

        # update Q-value, reward = d_score
        reward = state.getScore() - self.score 
        if len(self.lastStates) > 0:
            last_state = self.lastStates[-1]
            last_action = self.lastActions[-1]

            max_q = self.getMaxQ(curr_state, legal)
            self.updateQ(last_state, last_action, reward, max_q)

        # epsilon greedy
        if flipCoin(self.epsilon):
            action = choice(legal)
        else:
            action = self.greedyAction(curr_state, legal)

        # update attributes
        self.score = state.getScore()
        self.lastStates.append(curr_state)
        self.lastActions.append(action)

        return action

    def final(self, state):
        # update Q-value, reward = d_score
        reward = state.getScore() - self.score
        last_state = self.lastStates[-1]
        last_action = self.lastActions[-1]
        self.updateQ(last_state, last_action, reward, 0)

        # reset attributes
        self.score = 0
        self.lastStates = []
        self.lastActions = []

        # decrease epsilon during the training
        ep = 1 - self.episodesSoFar*1.0 / self.numTraining
        self.epsilon = (ep * 0.5)

        self.episodesSoFar += 1
        if self.episodesSoFar % 100 == 0:
            print('Completed %s runs of training' % self.episodesSoFar)

        if self.episodesSoFar >= self.numTraining:
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.alpha = 0
            self.epsilon = 0
