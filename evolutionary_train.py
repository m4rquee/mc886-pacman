from functools import partial

from pacman import *
from evolutionary.Population import Population

if __name__ == '__main__':
    """
    The main function called when evolutionary_train.py is run
    from the command line:

    > python evolutionary_train.py

    See the usage string for more details.

    > python evolutionary_train.py --help
    """
    args = readCommand(sys.argv[1:], False)  # Get game components based on input

    runner = partial(runGames, **args)
    checkpoint_file = args['checkpoint_file']
    population = Population(args['npop'], args['ngen'], args['numGames'],
                            runner, checkpoint_file=checkpoint_file)
    pop, stats, hof = population.evolve()

    input_str = input('Press enter to see best individuals running... ')
    population.tries = int(input_str or 1)
    for i, individual in enumerate(hof):
        print('\nRunning %d individual:' % i, individual)
        population.eval_individual(individual, True)
        input('Press enter to see next')

    # import cProfile
    # cProfile.run("runGames( **args )")
    pass
