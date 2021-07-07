from random import choice

from evolutionary.gpdef import PacmanSyntaxTree
from game import Agent


class EvolutionaryAgent(Agent):

    def __init__(self, name, index=0):
        super().__init__(index)
        self.syntax_tree = PacmanSyntaxTree(name)

    def getAction(self, state):
        return choice(state.getLegalPacmanActions())
