import time
from prepare_data import getTratedData
import classifier as clf

df = getTratedData()

x_train, x_test, y_train, y_test = clf.getTrainTestVariables(df)

population = clf.generatePopupation(10)

i = 1
"""
To do: 
- add a condition to stop the loop if the fitness doesn't change for 10 generations
- add a condition to stop the loop if the fitness is 1
- add mutation rate
- implement a crossover method
- add crossover rate
Generate charts to show the evolution of the fitness to make next steps
"""

while (True):
    print("Generation: ", i)
    population = clf.sortByFitness(
        population, x_train, x_test, y_train, y_test)
    if clf.checkIfFitnessDoesntChange(clf.calculateFitness(x_train, x_test, y_train, y_test, *population[0])):
        print("Best individual: ", population[0])
        print("Fitness: ", clf.calculateFitness(
            x_train, x_test, y_train, y_test, *population[0]))
        break
    parents1, parents2 = clf.parentSelection(
        population, x_train, x_test, y_train, y_test)
    # Crossover
    child1 = clf.crossover(population[0], population[1], 0.6)
    child2 = clf.crossover(population[1], population[0], 0.6)
    # Mutation
    child1 = clf.mutate(child1)
    child2 = clf.mutate(child2)
    # Sort by fitness
    population = clf.sortByFitness(
        population, x_train, x_test, y_train, y_test)
    # Replace the worst individuals
    population[-1] = child1
    population[-2] = child2
    # Sort by fitness
    population = clf.sortByFitness(
        population, x_train, x_test, y_train, y_test)
    # Check if fitness is 1
    bestFitness = clf.calculateFitness(
        x_train, x_test, y_train, y_test, *population[0])
    print("Best individual: ", population[0])
    print("Fitness: ", bestFitness)
    if(bestFitness == 1):
        break
    i += 1
    time.sleep(5)
