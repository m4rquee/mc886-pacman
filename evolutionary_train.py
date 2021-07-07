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
    args = readCommand(sys.argv[1:])  # Get game components based on input

    runner = partial(runGames, **args)
    population = Population(args['npop'], args['numGames'], runner)
    population.evolve()

    # import cProfile
    # cProfile.run("runGames( **args )")
    pass
