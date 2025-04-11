import operator
import math
import random

import numpy

from functools import partial

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp


# Main has 4 arguments (a, b, c, d) and can use one function (min)
pset = gp.PrimitiveSet("MAIN", 4)
pset.renameArguments(ARG0='a', ARG1='b', ARG2='c', ARG3='d')
pset.addPrimitive(min, 2)

# Not entirely sure what weights refers to, but it was in the template soooo ¯\_(ツ)_/¯
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Using half and half to generate trees of depth 2 because we don't want a very complicated solution
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# Disclaimer: ChatGPT wrote this function to generate data our data (50 quadruplets of integers 
# between -50 and 50)
def generate_test_cases(num_cases=50, low=-50, high=50):
    return [tuple(random.randint(low, high) for _ in range(4)) for _ in range(num_cases)]

test_cases = generate_test_cases()

def evalMin(individual):
    func = toolbox.compile(expr=individual)
    error = 0
    for a, b, c, d in test_cases:
        output = func(a, b, c, d)
        ground_truth = min(a, b, c, d)
        inputs = [a, b, c, d]
        rest = []
        for i in inputs:
            if i != output:
                rest.append(i)
        if output == ground_truth:
            error += 0
        else:
            error += abs(output - ground_truth) 
    return error,


toolbox.register("evaluate", evalMin)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

def main():
    random.seed(42) 
    pop = toolbox.population(n=300)  
    hof = tools.HallOfFame(1)       
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox,
                                   cxpb=0.5, mutpb=0.2, ngen=40,
                                   stats=mstats, halloffame=hof, verbose=True)

    for winner in hof:
        print(str(winner))

    return pop, log, hof

if __name__ == "__main__":
    main()