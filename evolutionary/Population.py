import random

import numpy as np
from deap import creator, base, gp, tools, algorithms

from evolutionary import EvolutionaryAgent
from evolutionary.gpdef import PacmanSyntaxTree


class Population:
    def eval_individual(self, individual):
        # Transform the tree expression in a callable function
        func = self.toolbox.compile(expr=individual)
        agent = EvolutionaryAgent(func)
        # tests the agent in the environment...
        return 0,

    def evolve(self):
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        algorithms.eaSimple(self.pop, self.toolbox, 0.5, 0.2, 40, stats,
                            halloffame=hof)
        return self.pop, stats, hof

    def __init__(self, n, seed=42):
        # Startup configurations:
        creator.create('FitnessMax', base.Fitness, weights=(1.0,))
        creator.create('Individual', gp.PrimitiveTree,
                       fitness=creator.FitnessMax)

        self.pset = PacmanSyntaxTree()
        self.toolbox = base.Toolbox()
        self.toolbox.register('expr', gp.genHalfAndHalf, pset=self.pset, min_=1,
                              max_=2)
        self.toolbox.register('individual', tools.initIterate,
                              creator.Individual, self.toolbox.expr)
        self.toolbox.register('population', tools.initRepeat, list,
                              self.toolbox.individual)
        self.toolbox.register('compile', gp.compile, pset=self.pset)

        self.toolbox.register("evaluate", self.eval_individual)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register("mutate", gp.mutUniform,
                              expr=self.toolbox.expr_mut, pset=self.pset)

        # Generates the population:
        random.seed(seed)
        self.pop = self.toolbox.population(n=n)
