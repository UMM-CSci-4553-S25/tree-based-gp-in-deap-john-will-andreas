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

#updated test cases to -100 through 100 to reflect the problem implementation
def generate_test_cases(num_cases=50, low=-100, high=100):
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

# run trial has been moved to a function so that independent trials can be run in series 
def run_single_trial(trial_id, verbose=False):
    """Run a single trial of the GP algorithm"""
    random.seed(42 + trial_id)  # Different seed for each trial
    
    # Initialize the population
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    
    # Set up statistics
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)
    
    if verbose:
        print(f"\n--- Starting Trial {trial_id+1} ---")
    
    # Run the evolutionary algorithm
    pop, log = algorithms.eaSimple(pop, toolbox,
                               cxpb=0.5, mutpb=0.2, ngen=40,
                               stats=mstats, halloffame=hof, 
                               verbose=verbose)
    
    # Get the best individual
    best_ind = hof[0]
    
    if verbose:
        print(f"\nTrial {trial_id+1} - Best individual:")
        print(str(best_ind))
        print(f"Fitness: {best_ind.fitness.values[0]}")
        print(f"Size: {len(best_ind)}")
    
    # Return the result
    return {
        "best_individual": best_ind,
        "best_fitness": best_ind.fitness.values[0],
        "best_size": len(best_ind),
        "log": log
    }

def plot_multitrial_statistics(trial_results):
    """Plot statistics across multiple trials"""
    # Create a figure with 3 subplots
    plt.figure(figsize=(15, 12))
    
    # Extract data
    best_fitnesses = [r["best_fitness"] for r in trial_results]
    best_sizes = [r["best_size"] for r in trial_results]
    
    # Calculate average fitness and size by generation across all trials
    gen_count = min(41, min([len(trial["log"]) for trial in trial_results]))  # Handle shorter logs
    avg_fitness_by_gen = []
    best_fitness_by_gen = []
    avg_size_by_gen = []
    
    for gen in range(gen_count):
        gen_fit_avg = []
        gen_fit_best = []
        gen_size_avg = []
        
        for trial in trial_results:
            log = trial["log"]
            if gen < len(log):
                # Access data through the logbook chapters directly
                try:
                    # Get fitness stats for this generation
                    gen_fit_avg.append(log.chapters["fitness"].select("avg")[gen])
                    gen_fit_best.append(log.chapters["fitness"].select("min")[gen])
                    gen_size_avg.append(log.chapters["size"].select("avg")[gen])
                except Exception as e:
                    print(f"Error accessing statistics for generation {gen}: {e}")
                    continue
        
        # Calculate means only if we have data
        if gen_fit_avg:
            avg_fitness_by_gen.append(sum(gen_fit_avg) / len(gen_fit_avg))
            best_fitness_by_gen.append(sum(gen_fit_best) / len(gen_fit_best))
            avg_size_by_gen.append(sum(gen_size_avg) / len(gen_size_avg))
    
    # Plot 1: Best fitness achieved in each trial
    plt.subplot(3, 1, 1)
    plt.bar(range(1, len(trial_results) + 1), best_fitnesses)
    plt.axhline(y=statistics.mean(best_fitnesses), color='r', linestyle='-', label=f'Mean: {statistics.mean(best_fitnesses):.2f}')
    plt.xlabel("Trial")
    plt.ylabel("Best Error (lower is better)")
    plt.title("Best Error Achieved in Each Trial")
    plt.legend()
    plt.grid(True, axis='y')
    
    # Plot 2: Average and best fitness by generation
    plt.subplot(3, 1, 2)
    plt.plot(range(len(avg_fitness_by_gen)), avg_fitness_by_gen, 'r-', label="Average Error")
    plt.plot(range(len(best_fitness_by_gen)), best_fitness_by_gen, 'b-', label="Best Error")
    plt.xlabel("Generation")
    plt.ylabel("Error (lower is better)")
    plt.title("Error Evolution Across Generations (Averaged Over All Trials)")
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Average tree size by generation
    plt.subplot(3, 1, 3)
    plt.plot(range(len(avg_size_by_gen)), avg_size_by_gen, 'g-', label="Average Size")
    plt.xlabel("Generation")
    plt.ylabel("Tree Size (nodes)")
    plt.title("Solution Size Evolution (Averaged Over All Trials)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("gp_multirun_statistics.png")
    plt.show()

def main():
    NUM_TRIALS = 10  # Number of independent runs
    
    print(f"Running {NUM_TRIALS} independent trials...")
    print("Test cases used for all trials:")
    for i, case in enumerate(test_cases[:5]):
        print(f"  Case {i+1}: {case} -> Min: {min(*case)}")
    print("  ...")
    
    # Run trials
    trial_results = []
    for i in range(NUM_TRIALS):
        result = run_single_trial(i, verbose=True)
        trial_results.append(result)
    
    # Calculate overall statistics
    print("\n===== OVERALL STATISTICS =====")
    best_fitnesses = [r["best_fitness"] for r in trial_results]
    best_sizes = [r["best_size"] for r in trial_results]
    
    print(f"Best Error Statistics:")
    print(f"  Mean: {statistics.mean(best_fitnesses):.2f}")
    print(f"  Std dev: {statistics.stdev(best_fitnesses) if len(best_fitnesses) > 1 else 0:.2f}")
    print(f"  Min: {min(best_fitnesses)}")
    print(f"  Max: {max(best_fitnesses)}")
    
    print(f"\nBest Solution Size Statistics:")
    print(f"  Mean: {statistics.mean(best_sizes):.2f}")
    print(f"  Std dev: {statistics.stdev(best_sizes) if len(best_sizes) > 1 else 0:.2f}")
    print(f"  Min: {min(best_sizes)}")
    print(f"  Max: {max(best_sizes)}")
    
    # Find the best solution across all trials
    best_trial = min(trial_results, key=lambda x: x["best_fitness"])
    best_ind = best_trial["best_individual"]
    print("\nBest overall individual:")
    print(str(best_ind))
    print(f"Fitness: {best_trial['best_fitness']}")
    print(f"Size: {best_trial['best_size']}")
    
    # Test the overall best individual on test cases
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
    
    # Plot statistics across all trials
    plot_multitrial_statistics(trial_results)
    
    # Also plot detailed statistics for the best run
    print("\nPlotting detailed statistics for the best run...")
    plot_statistics(best_trial["log"])
    
    return trial_results

if __name__ == "__main__":
    main()