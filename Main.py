import time
import random
from prepare_data import getTratedData
import classifier as clf
import warnings
warnings.filterwarnings("ignore")

df = getTratedData()

x_train, x_test, y_train, y_test = clf.getTrainTestVariables(df)

population = clf.generatePopupation(50)

i = 1

f = open("thanos3.txt", "a")

while (i < 100):
    print("Generation: ", i)
    population = clf.sortByFitness(
        population, x_train, x_test, y_train, y_test)
    if clf.checkIfFitnessDoesntChange(clf.calculateFitness(x_train, x_test, y_train, y_test, *population[0])):
        # Mutate the best individual
        population[0] = clf.mutate(population[0])

    # Delete 50% of population and replace it with new individuals (crossover and mutation)

    population = population[:25]
    for j in range(25):
        parent1 = population[random.randint(0, 24)]
        parent2 = population[random.randint(0, 24)]
        population.append(clf.crossover(parent1, parent2))

    # Sort by fitness
    population = clf.sortByFitness(
        population, x_train, x_test, y_train, y_test)
    # Check if fitness is 1
    bestFitness = clf.calculateFitness(
        x_train, x_test, y_train, y_test, *population[0])
    print("Best individual: ", population[0])
    print("Fitness: ", bestFitness)
    # Write in file and break line
    f.write(str(bestFitness) + "\n")

    if (bestFitness == 1):
        break
    i += 1

f.close()
