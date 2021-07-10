import pickle
from datetime import datetime
from random import random
from time import time

import numpy as np
from deap import creator, base, gp, tools, algorithms

from evolutionary import EvolutionaryAgent
from evolutionary.gpdef import PacmanSyntaxTree


def checkpoint_save(pop, gen_count, hof):
    ts = datetime.fromtimestamp(time()).isoformat()
    with open('%s.pkl' % ts, 'wb') as cp_file:
        pickle.dump(dict(pop=pop, gen_count=gen_count, hof=hof), cp_file)


class Population:
    WIN_WEIGHT = 10000
    MOVE_WEIGHT = 0.5

    def eval_individual(self, individual, show_gui=False):
        # Transform the tree expression in a callable function
        # print(individual)
        func = self.toolbox.compile(expr=individual)
        agent = EvolutionaryAgent(func)
        numTraining = 0 if show_gui else self.tries
        _, avg_score, win_rate, avg_move_count = \
            self.game_runner(pacman=agent, numTraining=numTraining)
        fitness = avg_score + Population.WIN_WEIGHT * win_rate
        if win_rate == 0: fitness -= Population.WIN_WEIGHT  # failure penalty
        return fitness, avg_move_count

    def random_mutation_operator(self, individual):
        """
        Randomly picks a replacement, insert, or shrink mutation.
        """
        roll = random()
        if roll <= 0.30:  # 30%
            return gp.mutUniform(individual, expr=self.toolbox.expr_mut, pset=self.pset)
        elif roll <= 0.50:  # 20%
            return gp.mutNodeReplacement(individual, pset=self.pset)
        elif roll <= 0.70:  # 20%
            return gp.mutInsert(individual, pset=self.pset)
        return gp.mutShrink(individual)  # 30%

    def evolve(self):
        self.hof = self.hof or tools.HallOfFame(3)
        if self.ngen == 0: return self.pop, None, self.hof
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats_fit.register('fit avg', np.mean)
        stats_fit.register('std', np.std)
        stats_fit.register('min', np.min)
        stats_fit.register('max', np.max)
        stats_moves = tools.Statistics(lambda ind: ind.fitness.values[1])
        stats_moves.register('moves avg', np.mean)
        mstats = tools.MultiStatistics(fitness=stats_fit, moves=stats_moves)

        for block in range(0, self.ngen, self.save_freq):
            algorithms.eaMuCommaLambda(self.pop, self.toolbox, self.n,
                                       self.lambda_, 0.5, 0.4,
                                       ngen=self.save_freq, stats=mstats,
                                       halloffame=self.hof, verbose=True)
            self.gen_count += self.save_freq
            checkpoint_save(self.pop, self.gen_count, self.hof)
        return self.pop, mstats, self.hof

    def __init__(self, n, ngen, tries, game_runner, save_freq=25,
                 checkpoint_file=None):
        # Startup configurations:
        self.n = n
        self.lambda_ = int(self.n * 1.5)
        self.ngen = ngen
        self.tries = tries
        self.game_runner = game_runner
        self.save_freq = min(ngen, save_freq)
        creator.create('FitnessMax', base.Fitness,
                       weights=(1.0, -Population.MOVE_WEIGHT))
        creator.create('Individual', gp.PrimitiveTree,
                       fitness=creator.FitnessMax)

        self.pset = PacmanSyntaxTree()
        self.toolbox = base.Toolbox()
        self.toolbox.register('expr', gp.genHalfAndHalf, pset=self.pset, min_=2,
                              max_=8)
        self.toolbox.register('individual', tools.initIterate,
                              creator.Individual, self.toolbox.expr)
        self.toolbox.register('population', tools.initRepeat, list,
                              self.toolbox.individual)
        self.toolbox.register('compile', gp.compile, pset=self.pset)

        self.toolbox.register('evaluate', self.eval_individual)
        tournsize = int(self.n * 0.1)
        self.toolbox.register('select', tools.selTournament, tournsize=tournsize)
        self.toolbox.register('mate', gp.cxOnePoint)
        self.toolbox.register('expr_mut', gp.genHalfAndHalf, min_=1, max_=5)
        self.toolbox.register('mutate', self.random_mutation_operator)

        self.gen_count = 0
        self.hof = None
        if checkpoint_file is not None:
            with open(checkpoint_file, 'rb') as cp_file:
                cp = pickle.load(cp_file)
            self.pop = cp['pop']
            for ind in self.pop:
                del ind.fitness.values
            self.gen_count = cp['gen_count']
            print('Resuming from last run of %d generation(s)' % self.gen_count)
            self.hof = cp['hof']
        else:
            self.pop = self.toolbox.population(n=n)  # generates a population
