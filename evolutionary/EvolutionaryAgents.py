import math
from functools import partial
from random import choice

import numpy as np

from game import Agent, Actions, Directions
from util import nearestPoint, manhattanDistance


def dist_to_next_pill(state, pos, current_dir):
    x, y = pos
    dx, dy = current_dir
    if dx == 0 and dy == 0: return -1  # the agents is idle
    k = 1  # k steps into the future
    while not state.hasWall(x + k * dx, y + k * dy):
        if state.hasFood(x + dx, y + dy): return k * (abs(dx) + abs(dy))
        k += 1
    return -1  # no pill in this direction


def dist_to_next_power_pill(state, pos):
    capsules = state.getCapsules()
    agent_dist = partial(manhattanDistance, pos)
    return min(map(agent_dist, capsules)) if len(capsules) > 0 else -1


def dist_to_ghost(state, pos):
    # Dump positons are added to call min with nonempty lists:
    edible, non_edible = [], []
    for ghost in state.getGhostStates():
        ghost_pos = ghost.getPosition()
        (edible if ghost.scaredTimer > 0 else non_edible).append(ghost_pos)
    agent_dist = partial(manhattanDistance, pos)
    eGhostX, eGhostY, neGhostX, neGhostY = -1, -1, -1, -1
    if len(edible) > 0:
        eGhostX, eGhostY = edible[np.argmin(map(agent_dist, edible))]
    if len(non_edible) > 0:
        neGhostX, neGhostY = non_edible[np.argmin(map(agent_dist, non_edible))]
    return eGhostX, eGhostY, len(edible), neGhostX, neGhostY, len(non_edible)


def next_junction(state, pos, current_dir):
    x, y = pos
    dx, dy = current_dir
    k = 1  # k steps into the future
    if dx == 0 and dy == 0: return -1, False  # the agents is idle
    ghost_before_junction = False
    ghosts = [nearestPoint(pos) for pos in state.getGhostPositions()]
    while not state.hasWall(x + k * dx, y + k * dy):
        if (x + k * dx, y + k * dy) in ghosts: ghost_before_junction = True
        k += 1
    return k * (abs(dx) + abs(dy)), ghost_before_junction


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
        DistToNextPill = dist_to_next_pill(state, nearest, current_dir)
        DistToNextPowerPill = dist_to_next_power_pill(state, nearest)
        EdibleGhostX, EdibleGhostY, GdEdibleGhostCount, \
        NonEdibleGhostX, NonEdibleGhostY, GdNonEdibleGhostCount = \
            dist_to_ghost(state, nearest)
        DistToNextJunction, GhostBeforeJunction = \
            next_junction(state, nearest, current_dir)
        GdPillCount = state.getNumFood()
        GdPowerPillCount = len(state.getCapsules())
        Score = state.getScore()
        DirectionX, DirectionY = current_dir

        # Condensate all gathered information:
        condensed_state = {'DistToNextPill': DistToNextPill,
                           'DistToNextPowerPill': DistToNextPowerPill,
                           'EdibleGhostX': EdibleGhostX,
                           'EdibleGhostY': EdibleGhostY,
                           'NonEdibleGhostX': NonEdibleGhostX,
                           'NonEdibleGhostY': NonEdibleGhostY,
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

        legal_actions = map(Actions.directionToVector, state.getLegalPacmanActions())

        q_values = [(-math.inf, Actions.directionToVector(Directions.STOP))]
        for X, Y in legal_actions:
            try:
                q_values.append((partial_func(ActionX=X, ActionY=Y), (X, Y)))
            except Exception as excep:
                print('\nError while evaluation tree:')
                print(excep)
                print('ActionX=%d, ActionY=%d, condensed_state=%s\n' % (X, Y, str(condensed_state)))
        best_q = max(q_values)[0]
        action = choice([pair[1] for pair in q_values if pair[0] == best_q])
        return Actions.vectorToDirection(action)
