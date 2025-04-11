import random
from deap import base, creator, tools

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

IND_SIZE = 4

toolbox = base.Toolbox()
toolbox.register("attribute", random.randint, -100, 100)
toolbox.register(
    "individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=IND_SIZE
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evaluate(individual):
    return (sum(individual),)


toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=20, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)


def main():
    pop = toolbox.population(n=50)
    CXPB, MUTPB, NGEN = 0.5, 0.2, 40

    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Track best individual
    best_ind = tools.selBest(pop, 1)[0]
    print(f"Gen 0: Best = {best_ind}, Fitness = {best_ind.fitness.values[0]}")

    for g in range(NGEN):
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                # Round to integers after Gaussian mutation
                for i in range(len(mutant)):
                    mutant[i] = int(round(mutant[i]))
                    # Ensure values stay within range
                    mutant[i] = max(-100, min(100, mutant[i]))
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Display progress
        if (g + 1) % 10 == 0 or g == NGEN - 1:
            best_ind = tools.selBest(pop, 1)[0]
            print(
                f"Gen {g + 1}: Best = {best_ind}, Fitness = {best_ind.fitness.values[0]}"
            )

    return pop


if __name__ == "__main__":
    result = main()
    best_ind = tools.selBest(result, 1)[0]
    print(f"Best individual: {best_ind}")
    print(f"Fitness: {best_ind.fitness.values[0]}")
