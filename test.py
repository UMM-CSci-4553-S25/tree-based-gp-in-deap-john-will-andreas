import random
from deap import base, creator, tools, gp

# Create fitness and individual
creator.create("FitnessMin", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

# Define primitive set (operations, inputs)
pset = gp.PrimitiveSet("MAIN", 4)  # 4 inputs: a, b, c, d
pset.addPrimitive(min, 2)           # only minimum of two values
pset.renameArguments(ARG0="a", ARG1="b", ARG2="c", ARG3="d")

toolbox = base.Toolbox()
# Tree initialization
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Fixed training cases - 10 sets of 4 random integers
TRAINING_CASES = []
for _ in range(10):
    case = [random.randint(-100, 100) for _ in range(4)]
    TRAINING_CASES.append((case, min(case)))

def evaluate(individual):
    # Transform the tree expression into a callable function
    func = gp.compile(individual, pset)
    
    # Evaluate the individual on each test case
    correct = 0
    for inputs, expected in TRAINING_CASES:
        try:
            result = func(*inputs)
            if result == expected:
                correct += 1
        except Exception:
            # Handle potential errors (division by zero, etc.)
            pass
    
    return correct,  # Return as tuple

# Register the evolutionary operators
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.register("evaluate", evaluate)

def main():
    pop = toolbox.population(n=300)
    CXPB, MUTPB, NGEN = 0.5, 0.2, 40
    
    # Print the training cases
    print("Training Cases:")
    for i, (inputs, expected) in enumerate(TRAINING_CASES):
        print(f"Case {i+1}: {inputs} -> Min: {expected}")
    
    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    # Track best individual
    best_ind = tools.selBest(pop, 1)[0]
    print(f"Gen 0: Best fitness = {best_ind.fitness.values[0]}/10")
    
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
        
        # Display progress
        if (g+1) % 10 == 0 or g == NGEN-1:
            best_ind = tools.selBest(pop, 1)[0]
            print(f"Gen {g+1}: Best fitness = {best_ind.fitness.values[0]}/10")

    return pop

if __name__ == "__main__":
    result = main()
    best_ind = tools.selBest(result, 1)[0]
    print("\nBest individual:")
    print(best_ind)
    print(f"Fitness: {best_ind.fitness.values[0]}/10")
    
    # Test the best individual on training cases
    func = gp.compile(best_ind, pset)
    print("\nResults on training cases:")
    for i, (inputs, expected) in enumerate(TRAINING_CASES):
        try:
            result = func(*inputs)
            print(f"Case {i+1}: {inputs} -> Expected: {expected}, Got: {result}")
        except Exception as e:
            print(f"Case {i+1}: {inputs} -> Error: {e}")