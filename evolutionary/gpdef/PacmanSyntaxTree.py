import math
import operator

from deap.gp import PrimitiveSetTyped


def safe_div(a, b): return 0 if b == 0 else (a / b)
def safe_floordiv(a, b): return 0 if b == 0 else (a // b)
def safe_mod(a, b): return 0 if b == 0 else (a % b)
def mean(a, b): return (a + b) / 2.0
def relu(x): return max(0, x)
def if_then_else(cond, a, b): return a if cond else b
def dist(a, b, c, d): return abs(a - c) + abs(b - d)


class PacmanSyntaxTree(PrimitiveSetTyped):
    IN_TYPE_MAP = {'DistToNextPill': float, 'DistToNextPowerPill': float,
                   'EdibleGhostX': float, 'EdibleGhostY': float,
                   'NonEdibleGhostX': float, 'NonEdibleGhostY': float,
                   'DistToNextJunction': float, 'GhostBeforeJunction': bool,
                   'GdPillCount': float, 'GdPowerPillCount': float,
                   'GdEdibleGhostCount': float, 'GdNonEdibleGhostCount': float,
                   'Score': float, 'DirectionX': float, 'DirectionY': float,
                   'PosX': float, 'PosY': float,
                   'ActionX': float, 'ActionY': float}
    FLOAT_CONSTS = [-1.0, 0.0, 0.1, 0.5, 1.0, 2.0, math.pi, 5.0, 10.0]

    def __init__(self, name='PacmanSyntaxTree'):
        in_types = PacmanSyntaxTree.IN_TYPE_MAP.values()
        super().__init__(name, in_types, float, prefix='')

        # Terminals:
        for float_const in PacmanSyntaxTree.FLOAT_CONSTS:
            self.addTerminal(float_const, float)
        self.addTerminal(True, bool)
        self.addTerminal(False, bool)

        # Bool operations:
        self.addPrimitive(operator.and_, [bool, bool], bool)
        self.addPrimitive(operator.or_, [bool, bool], bool)
        self.addPrimitive(operator.xor, [bool, bool], bool)
        self.addPrimitive(operator.not_, [bool], bool)

        # Mathematical operations:
        self.addPrimitive(operator.add, [float, float], float)
        self.addPrimitive(operator.sub, [float, float], float)
        self.addPrimitive(operator.mul, [float, float], float)
        self.addPrimitive(safe_div, [float, float], float)
        # self.addPrimitive(safe_mod, [float, float], float)
        # self.addPrimitive(safe_floordiv, [float, float], float)
        self.addPrimitive(operator.abs, [float], float)
        self.addPrimitive(operator.neg, [float], float)
        self.addPrimitive(max, [float, float], float)
        self.addPrimitive(min, [float, float], float)
        self.addPrimitive(mean, [float, float], float)
        self.addPrimitive(relu, [float], float)
        # self.addPrimitive(math.cos, [float], float)
        # self.addPrimitive(math.sin, [float], float)
        # self.addPrimitive(math.ceil, [float], float)
        self.addPrimitive(math.floor, [float], float)

        # Ternary primitives:
        self.addPrimitive(if_then_else, [bool, float, float], float)

        # Distance primitive:
        self.addPrimitive(dist, [float, float, float, float], float)

        # Comparison operations:
        self.addPrimitive(operator.lt, [float, float], bool)
        # self.addPrimitive(operator.le, [float, float], bool)
        self.addPrimitive(operator.eq, [float, float], bool)
        self.addPrimitive(operator.ne, [float, float], bool)
        # self.addPrimitive(operator.ge, [float, float], bool)
        # self.addPrimitive(operator.gt, [float, float], bool)

        # Use more readable names:
        new_names = PacmanSyntaxTree.IN_TYPE_MAP.keys()
        old_names = map(str, range(len(new_names)))
        self.renameArguments(**dict(zip(old_names, new_names)))
