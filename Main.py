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
- algorithm flow: generate population > sort by fitness > check if fitness doesn't change > if it doesn't change, stop the loop > if it changes, parents selection > crossover > mutate the population > if the fitness is 1, stop the loop > if the fitness is not 1, repeat the loop. Seems like function otimization problem.

Generate charts to show the evolution of the fitness to make next steps
"""

while(True):
    print("Generation: ", i)
    population = clf.sortByFitness(
        population, x_train, x_test, y_train, y_test)
    if clf.checkIfFitnessDoesntChange(clf.calculateFitness(x_train, x_test, y_train, y_test, *population[0])):
        print("Best individual: ", population[0])
        print("Fitness: ", clf.calculateFitness(
            x_train, x_test, y_train, y_test, *population[0]))
        break
    parents1, parents2 = clf.parentSelection(population)

    print("Best individual: ", population[0])
    print("Fitness: ", clf.calculateFitness(
        x_train, x_test, y_train, y_test, *population[0]))
    i += 1
    time.sleep(5)
