import math
from random import choice
from functools import partial

import numpy as np

from game import Agent, Actions, Directions
from util import Counter, flipCoin, nearestPoint, manhattanDistance

def next_pill(state, pos):
    agent_dist = partial(manhattanDistance, pos)
    if state.getNumFood() == 0: return -1, -1  # no pill left
    min_dist, pill_pos = math.inf, pos
    for i, l in enumerate(state.getFood()):
        for j, f in enumerate(l):
            aux = agent_dist((i, j))
            if f and aux < min_dist:
                min_dist = aux
                pill_pos = (i, j)
    return pill_pos

def next_power_pill(state, pos):
    capsules = state.getCapsules()
    agent_dist = partial(manhattanDistance, pos)
    dists = map(agent_dist, capsules)
    return capsules[np.argmin(dists)] if len(capsules) > 0 else (-1, -1)

def closest_ghosts(state, pos):
    # Dump positons are added to call min with nonempty lists:
    edible, non_edible = [], []
    for ghost in state.getGhostStates():
        ghost_pos = ghost.getPosition()
        (edible if ghost.scaredTimer > 0 else non_edible).append(ghost_pos)
    agent_dist = partial(manhattanDistance, pos)
    eGhostX, eGhostY, neGhostX, neGhostY = -1, -1, -1, -1
    eGhostDist, neGhostDist = -1, -1
    if len(edible) > 0:
        dists = list(map(agent_dist, edible))
        eGhostX, eGhostY = edible[np.argmin(dists)]
        eGhostDist = min(dists)
    if len(non_edible) > 0:
        dists = list(map(agent_dist, non_edible))
        neGhostX, neGhostY = non_edible[np.argmin(dists)]
        neGhostDist = min(dists)
    total = len(edible) + len(non_edible)
    return eGhostX, eGhostY, eGhostDist, len(edible) / total, \
           neGhostX, neGhostY, neGhostDist, len(non_edible) / total

def next_wall(state, pos, current_dir):
    x, y = pos
    dx, dy = current_dir
    k = 1  # k steps into the future
    if dx == 0 and dy == 0: return -1, 0  # the agents is idle
    ghost_before_wall = 0
    ghosts = [nearestPoint(pos) for pos in state.getGhostPositions()]
    while not state.hasWall(x + k * dx, y + k * dy):
        if (x + k * dx, y + k * dy) in ghosts: ghost_before_wall = 1
        k += 1
    return k, ghost_before_wall

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

        self.MoveCount = 0
        self.food_total = 0
        self.capsule_total = 0

    def featureExtractor(self, state):
        feature = None
        
        # Get higher level state attributes:
        pos = state.getPacmanPosition()
        PosX, PosY = pos
        nearest = nearestPoint(pos)
        
        # Integer direction vector:
        current_dir = Actions.directionToVector(state.getPacmanState().configuration.direction, 1)
        NextPillX, NextPillY = next_pill(state, nearest)
        NextPowerPillX, NextPowerPillY = next_power_pill(state, nearest)
        EdibleGhostX, EdibleGhostY, EdibleGhostDist, GdEdibleGhostCount, \
        NonEdibleGhostX, NonEdibleGhostY, NonEdibleGhostDist, GdNonEdibleGhostCount = \
            closest_ghosts(state, nearest)
        DistToNextJunction, GhostBeforeJunction = \
            next_wall(state, nearest, current_dir)
        GdPillCount = state.getNumFood() / self.food_total
        GdPowerPillCount = len(state.getCapsules()) / self.capsule_total
        Score = state.getScore()
        DirectionX, DirectionY = current_dir

        # Condensate all gathered information:
        condensed_state = {'NextPillX': NextPillX, 'NextPillY': NextPillY,
                           'NextPowerPillX': NextPowerPillX,
                           'NextPowerPillY': NextPowerPillY,
                           'EdibleGhostX': EdibleGhostX,
                           'EdibleGhostY': EdibleGhostY,
                           'EdibleGhostDist': EdibleGhostDist,
                           'NonEdibleGhostX': NonEdibleGhostX,
                           'NonEdibleGhostY': NonEdibleGhostY,
                           'NonEdibleGhostDist': NonEdibleGhostDist,
                           'DistToNextJunction': DistToNextJunction,
                           'GhostBeforeJunction': GhostBeforeJunction,
                           'GdPillCount': GdPillCount,
                           'GdPowerPillCount': GdPowerPillCount,
                           'GdEdibleGhostCount': GdEdibleGhostCount,
                           'GdNonEdibleGhostCount': GdNonEdibleGhostCount,
                           'Score': Score, 'DirectionX': DirectionX,
                           'DirectionY': DirectionY,
                           'MoveCount': self.MoveCount,
                           'PosX': PosX, 'PosY': PosY}

        feature = hash(frozenset(condensed_state.items()))

        return feature

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

    def registerInitialState(self, state):
        self.food_total = state.getNumFood()
        self.capsule_total = len(state.getCapsules())

    def getAction(self, state):
        # The data we have about the state of the game
        # the legal action of this state
        legal = state.getLegalPacmanActions()

        # Avoid stopping
        if len(legal) > 1 and Directions.STOP in legal:
            legal.remove(Directions.STOP)

        curr_state = self.featureExtractor(state)

        # update Q-value, reward = d_score
        reward = state.getScore() - self.score 
        if len(self.lastStates) > 0:
            last_state = self.lastStates[-1]
            last_action = self.lastActions[-1]

            # general rewarding for positive delta score
            if reward > 0:
                reward += 500
            # Punish stalling and reward moving forward
            if len(self.lastActions) >= 2 and last_action == Directions.REVERSE[self.lastActions[-2]]:
                reward -= 1000
            if len(self.lastActions) >= 2 and last_action == self.lastActions[-2]:
                reward += 100

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
