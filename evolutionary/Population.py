import numpy as np
from deap import creator, base, gp, tools, algorithms

from evolutionary import EvolutionaryAgent
from evolutionary.gpdef import PacmanSyntaxTree


class Population:
    WIN_WEIGHT = 1000

    def eval_individual(self, individual, show_gui=False):
        # Transform the tree expression in a callable function
        # print(individual)
        func = self.toolbox.compile(expr=individual)
        agent = EvolutionaryAgent(func)
        numTraining = 0 if show_gui else self.tries
        _, avg_score, win_rate = \
            self.game_runner(pacman=agent, numTraining=numTraining)
        return avg_score + Population.WIN_WEIGHT * win_rate,

    def evolve(self):
        hof = tools.HallOfFame(3)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        algorithms.eaSimple(self.pop, self.toolbox, 0.1, 0.1, self.ngen, stats,
                            halloffame=hof)
        return self.pop, stats, hof

    def __init__(self, n, ngen, tries, game_runner):
        # Startup configurations:
        self.ngen = ngen
        self.tries = tries
        self.game_runner = game_runner
        creator.create('FitnessMax', base.Fitness, weights=(1.0,))
        creator.create('Individual', gp.PrimitiveTree,
                       fitness=creator.FitnessMax)

        self.pset = PacmanSyntaxTree()
        self.toolbox = base.Toolbox()
        self.toolbox.register('expr', gp.genHalfAndHalf, pset=self.pset, min_=2,
                              max_=5)
        self.toolbox.register('individual', tools.initIterate,
                              creator.Individual, self.toolbox.expr)
        self.toolbox.register('population', tools.initRepeat, list,
                              self.toolbox.individual)
        self.toolbox.register('compile', gp.compile, pset=self.pset)

        self.toolbox.register("evaluate", self.eval_individual)
        self.toolbox.register("select", tools.selTournament, tournsize=10)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register("mutate", gp.mutUniform,
                              expr=self.toolbox.expr_mut, pset=self.pset)

        # Generates the population:
        self.pop = self.toolbox.population(n=n)
