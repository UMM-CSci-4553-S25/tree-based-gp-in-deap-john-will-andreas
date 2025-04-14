import operator
import math
import random

import numpy
import matplotlib.pyplot as plt
import statistics

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

def plot_statistics(log):
    """Plot the statistics from the logbook"""
    gen = log.select("gen")
    
    # Extract fitness statistics (note: negating to show improvement as upward trend)
    fit_mins = [-x for x in log.chapters["fitness"].select("min")]
    fit_avgs = [-x for x in log.chapters["fitness"].select("avg")]
    fit_stds = log.chapters["fitness"].select("std")
    
    # Extract size statistics
    size_avgs = log.chapters["size"].select("avg")
    size_mins = log.chapters["size"].select("min")
    size_maxs = log.chapters["size"].select("max")
    
    # Create a new figure with 2 subplots
    plt.figure(figsize=(12, 8))
    
    # Plot fitness evolution
    plt.subplot(2, 1, 1)
    plt.plot(gen, fit_mins, "b-", label="Best Error")
    plt.plot(gen, fit_avgs, "r-", label="Average Error")
    plt.fill_between(gen, [avg - std for avg, std in zip(fit_avgs, fit_stds)],
                     [avg + std for avg, std in zip(fit_avgs, fit_stds)],
                     alpha=0.2, color="r")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (lower is better)")
    plt.title("Fitness Evolution")
    plt.legend(loc="best")
    plt.grid(True)
    
    # Plot size evolution
    plt.subplot(2, 1, 2)
    plt.plot(gen, size_avgs, "g-", label="Average Size")
    plt.plot(gen, size_mins, "b-", label="Minimum Size")
    plt.plot(gen, size_maxs, "r-", label="Maximum Size")
    plt.xlabel("Generation")
    plt.ylabel("Tree Size (nodes)")
    plt.title("Size Evolution")
    plt.legend(loc="best")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("gp_statistics.png")
    plt.show()

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

    # Print the best individual 
    best_ind = hof[0]
    print("\nBest individual:")
    print(str(best_ind))
    print(f"Fitness: {best_ind.fitness.values[0]}")
    print(f"Size: {len(best_ind)}")

    # Print statistics about the run
    print("\n===== STATISTICS =====")
    # Get final generation statistics
    last_gen = log.select("gen")[-1]
    last_stats = log.chapters["fitness"].select("min", "avg", "std")[-1]
    last_size_stats = log.chapters["size"].select("min", "avg", "max")[-1]
    
    print(f"Final Generation: {last_gen}")
    print(f"Best Error: {last_stats[0]}")
    print(f"Average Error: {last_stats[1]:.2f}")
    print(f"Std Dev: {last_stats[2]:.2f}")
    
    print(f"\nSolution Size Statistics:")
    print(f"  Min Size: {last_size_stats[0]}")
    print(f"  Avg Size: {last_size_stats[1]:.2f}")
    print(f"  Max Size: {last_size_stats[2]}")

    # Test the best individual on test cases
    func = gp.compile(best_ind, pset)
    print("\nResults on select test cases:")
    for i, (a, b, c, d) in enumerate(test_cases[:10]):  # Show first 10 cases
        try:
            result = func(a, b, c, d)
            expected = min(a, b, c, d)
            status = "✓" if result == expected else "✗"
            print(f"Case {i+1}: {(a, b, c, d)} -> Expected: {expected}, Got: {result} {status}")
        except Exception as e:
            print(f"Case {i+1}: {(a, b, c, d)} -> Error: {e}")
    
    # Plot the statistics
    plot_statistics(log)

    return pop, log, hof

if __name__ == "__main__":
    main()