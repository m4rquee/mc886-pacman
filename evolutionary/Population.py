import pickle
from datetime import datetime
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
    MOVE_WEIGHT = 0.1

    def eval_individual(self, individual, show_gui=False):
        # Transform the tree expression in a callable function
        # print(individual)
        func = self.toolbox.compile(expr=individual)
        agent = EvolutionaryAgent(func)
        numTraining = 0 if show_gui else self.tries
        _, avg_score, win_rate, avg_move_count = \
            self.game_runner(pacman=agent, numTraining=numTraining)
        fitness = avg_score + Population.WIN_WEIGHT * win_rate
        return fitness, avg_move_count

    def evolve(self):
        self.hof = self.hof or tools.HallOfFame(3)
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats_fit.register('fit avg', np.mean)
        stats_fit.register('std', np.std)
        stats_fit.register('min', np.min)
        stats_fit.register('max', np.max)
        stats_moves = tools.Statistics(lambda ind: ind.fitness.values[1])
        stats_moves.register('moves avg', np.mean)
        mstats = tools.MultiStatistics(fitness=stats_fit, moves=stats_moves)

        for block in range(0, self.ngen, self.save_freq):
            algorithms.eaMuPlusLambda(self.pop, self.toolbox, self.n, 10, 0.8,
                                      0.1, self.save_freq, stats=mstats,
                                      halloffame=self.hof, verbose=True)
            self.gen_count += self.save_freq
            checkpoint_save(self.pop, self.gen_count, self.hof)
        return self.pop, mstats, self.hof

    def __init__(self, n, ngen, tries, game_runner, save_freq=10,
                 checkpoint_file=None):
        # Startup configurations:
        self.n = n
        self.ngen = ngen
        self.tries = tries
        self.game_runner = game_runner
        self.save_freq = min(ngen, save_freq)
        creator.create('FitnessMax', base.Fitness,
                       weights=(1.0, Population.MOVE_WEIGHT))
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

        self.toolbox.register('evaluate', self.eval_individual)
        self.toolbox.register('select', tools.selTournament, tournsize=10)
        self.toolbox.register('mate', gp.cxOnePoint)
        self.toolbox.register('expr_mut', gp.genFull, min_=0, max_=2)
        self.toolbox.register('mutate', gp.mutUniform,
                              expr=self.toolbox.expr_mut, pset=self.pset)

        self.gen_count = 0
        self.hof = None
        if checkpoint_file is not None:
            with open(checkpoint_file, 'rb') as cp_file:
                cp = pickle.load(cp_file)
            self.pop = cp['pop']
            self.gen_count = cp['gen_count']
            print('Resuming from last run of %d generation(s)' % self.gen_count)
            self.hof = cp['hof']
        else:
            self.pop = self.toolbox.population(n=n)  # generates a population
