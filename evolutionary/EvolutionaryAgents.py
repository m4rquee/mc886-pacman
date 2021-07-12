import math
from functools import partial
from random import choice, choices

import numpy as np

from game import Agent, Actions, Directions
from util import nearestPoint, manhattanDistance


def next_pill(state, pos):
    agent_dist = partial(manhattanDistance, pos)
    if state.getNumFood() == 0: return -1, -1  # no pill left
    min_dist, pill_pos = math.inf, pos
    for i, l in enumerate(state.getFood()):
        for j, f in enumerate(l):
            if f and (aux := agent_dist((i, j))) < min_dist:
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
    return eGhostX, eGhostY, eGhostDist, len(edible), \
           neGhostX, neGhostY, neGhostDist, len(non_edible)


def next_wall(state, pos, current_dir):
    x, y = pos
    dx, dy = current_dir
    k = 1  # k steps into the future
    if dx == 0 and dy == 0: return -1, False  # the agents is idle
    ghost_before_wall = False
    ghosts = [nearestPoint(pos) for pos in state.getGhostPositions()]
    while not state.hasWall(x + k * dx, y + k * dy):
        if (x + k * dx, y + k * dy) in ghosts: ghost_before_wall = True
        k += 1
    return k, ghost_before_wall


class EvolutionaryAgent(Agent):
    def __init__(self, tree_func, index=0):
        super().__init__(index)
        self.tree_func = tree_func

    def getAction(self, state):
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
        GdPillCount = state.getNumFood()
        GdPowerPillCount = len(state.getCapsules())
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
                           'PosX': PosX, 'PosY': PosY}
        partial_func = partial(self.tree_func, **condensed_state)

        legal_actions = state.getLegalPacmanActions()
        if len(legal_actions) == 1:
            return legal_actions.pop()  # the only option
        elif Directions.STOP in legal_actions:
            legal_actions.remove(Directions.STOP)  # avoid stopping

        q_values = []
        for X, Y in map(Actions.directionToVector, legal_actions):
            try:
                q_values.append(partial_func(ActionX=X, ActionY=Y))
            except Exception as excep:
                print('\nError while evaluation tree:')
                print(excep)
                print('ActionX=%d, ActionY=%d, condensed_state=%s\n' % (X, Y, str(condensed_state)))
                return choice(legal_actions)  # safety measure
        return choices(legal_actions, q_values)[0]
