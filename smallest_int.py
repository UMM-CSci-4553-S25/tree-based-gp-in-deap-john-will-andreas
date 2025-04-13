import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, gp
import statistics
import multiprocessing

# Create fitness and individual
creator.create("FitnessMin", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

# Define primitive set (operations, inputs)
pset = gp.PrimitiveSet("MAIN", 4)  # 4 inputs: a, b, c, d
pset.addPrimitive(min, 2)           # only minimum of two values
pset.renameArguments(ARG0="a", ARG1="b", ARG2="c", ARG3="d")

toolbox = base.Toolbox()
# Tree initialization
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Fixed training cases - 10 sets of 4 random integers
def generate_training_cases(num_cases=10, range_min=-100, range_max=100):
    cases = []
    for _ in range(num_cases):
        case_input = [random.randint(range_min, range_max) for _ in range(4)]
        cases.append((case_input, min(case_input)))
    return cases

def evaluate(individual, training_cases):
    # Transform the tree expression into a callable function
    func = gp.compile(individual, pset)
    
    # Evaluate the individual on each test case
    correct = 0
    for inputs, expected in training_cases:
        try:
            result = func(*inputs)
            if result == expected:
                correct += 1
        except Exception:
            # Handle potential errors
            pass
    
    return correct,  # Return as tuple

# Register the evolutionary operators
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# Set up statistics to collect
stats_fit = tools.Statistics(lambda ind: ind.fitness.values[0])
stats_size = tools.Statistics(len)
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
mstats.register("avg", np.mean)
mstats.register("std", np.std)
mstats.register("min", np.min)
mstats.register("max", np.max)

def run_single_trial(trial_id, training_cases, verbose=False):
    """Run a single trial of the GP algorithm"""
    random.seed(42 + trial_id)  # Different seed for each trial
    
    # Create a new evaluation function with our training cases
    toolbox.register("evaluate", evaluate, training_cases=training_cases)
    
    # Initialize the population
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    
    # Parameters
    CXPB, MUTPB, NGEN = 0.5, 0.2, 40
    
    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    # Track statistics by generation
    gen_stats = []
    
    # Initial statistics
    record = mstats.compile(pop)
    gen_stats.append(record)
    
    if verbose:
        print(f"Trial {trial_id+1} - Gen 0: Best fitness = {tools.selBest(pop, 1)[0].fitness.values[0]}/10")
    
    for g in range(NGEN):
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring
        
        # Update hall of fame and statistics
        hof.update(pop)
        record = mstats.compile(pop)
        gen_stats.append(record)
        
        # Display progress
        if verbose and ((g+1) % 10 == 0 or g == NGEN-1):
            best_ind = tools.selBest(pop, 1)[0]
            print(f"Trial {trial_id+1} - Gen {g+1}: Best fitness = {best_ind.fitness.values[0]}/10")
    
    # Return the best individual and all statistics
    best_ind = tools.selBest(pop, 1)[0]
    return {
        "best_individual": best_ind,
        "best_fitness": best_ind.fitness.values[0], 
        "best_size": len(best_ind),
        "gen_stats": gen_stats
    }

def main():
    NUM_TRIALS = 10  # Number of independent runs
    NUM_CASES = 10   # Number of training cases
    
    print(f"Running {NUM_TRIALS} independent trials...")
    
    # Generate the training cases once for all trials
    training_cases = generate_training_cases(NUM_CASES)
    
    # Print the training cases
    print("Training Cases:")
    for i, (inputs, expected) in enumerate(training_cases):
        print(f"Case {i+1}: {inputs} -> Min: {expected}")
    
    # Run trials (optional: use multiprocessing pool for parallel runs)
    results = []
    for i in range(NUM_TRIALS):
        trial_result = run_single_trial(i, training_cases, verbose=True)
        results.append(trial_result)
        
        # Print summary after each trial
        print(f"\nTrial {i+1} Summary:")
        print(f"Best individual: {trial_result['best_individual']}")
        print(f"Best fitness: {trial_result['best_fitness']}/{NUM_CASES}")
        print(f"Individual size: {trial_result['best_size']}")
        print("-" * 50)
    
    # Analyze results across all trials
    print("\n===== OVERALL STATISTICS =====")
    best_fitnesses = [r["best_fitness"] for r in results]
    best_sizes = [r["best_size"] for r in results]
    
    print(f"Best Fitness Statistics:")
    print(f"  Mean: {statistics.mean(best_fitnesses):.2f}")
    print(f"  Std dev: {statistics.stdev(best_fitnesses) if len(best_fitnesses) > 1 else 0:.2f}")
    print(f"  Min: {min(best_fitnesses)}")
    print(f"  Max: {max(best_fitnesses)}")
    
    print(f"\nBest Solution Size Statistics:")
    print(f"  Mean: {statistics.mean(best_sizes):.2f}")
    print(f"  Std dev: {statistics.stdev(best_sizes) if len(best_sizes) > 1 else 0:.2f}")
    print(f"  Min: {min(best_sizes)}")
    print(f"  Max: {max(best_sizes)}")
    
    # Calculate average performance by generation across all trials
    avg_fitness_by_gen = []
    best_fitness_by_gen = []
    avg_size_by_gen = []
    
    for gen in range(41):  # 0 to 40 generations
        gen_fit_avg = []
        gen_fit_best = []
        gen_size_avg = []
        
        for trial in results:
            stats = trial["gen_stats"][gen]
            gen_fit_avg.append(stats["fitness"]["avg"])
            gen_fit_best.append(stats["fitness"]["max"])
            gen_size_avg.append(stats["size"]["avg"])
        
        avg_fitness_by_gen.append(statistics.mean(gen_fit_avg))
        best_fitness_by_gen.append(statistics.mean(gen_fit_best))
        avg_size_by_gen.append(statistics.mean(gen_size_avg))
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(avg_fitness_by_gen, label="Average Fitness")
    plt.plot(best_fitness_by_gen, label="Best Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (correct cases)")
    plt.title("Fitness Evolution Across Generations")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(avg_size_by_gen, label="Average Solution Size")
    plt.xlabel("Generation")
    plt.ylabel("Tree Size (nodes)")
    plt.title("Solution Size Evolution Across Generations")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("gp_statistics.png")
    plt.show()
    
    # Find and show the best solution from all trials
    best_trial = max(results, key=lambda x: x["best_fitness"])
    best_overall = best_trial["best_individual"]
    
    print("\nBest Overall Solution:")
    print(best_overall)
    print(f"Fitness: {best_trial['best_fitness']}/{NUM_CASES}")
    print(f"Size: {best_trial['best_size']}")
    
    # Test the best individual on training cases
    func = gp.compile(best_overall, pset)
    print("\nResults on training cases:")
    for i, (inputs, expected) in enumerate(training_cases):
        try:
            result = func(*inputs)
            print(f"Case {i+1}: {inputs} -> Expected: {expected}, Got: {result}")
        except Exception as e:
            print(f"Case {i+1}: {inputs} -> Error: {e}")

if __name__ == "__main__":
    main()