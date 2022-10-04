import pandas as pd
from prepare_data import getTratedData
import classifier as clf

df = getTratedData()

x_train, x_test, y_train, y_test = clf.getTrainTestVariables(df)

initialPopulation = clf.generatePopupation(10)

i = 1
"""
To do: 
- add a condition to stop the loop if the fitness doesn't change for 10 generations
- add a condition to stop the loop if the fitness is 1
- add mutation rate
- implement a crossover method
- add crossover rate
- algorithm flow: generate population > sort by fitness > check if fitness doesn't change > if it doesn't change, stop the loop > if it changes, parents selection > crossover > mutate the population > if the fitness is 1, stop the loop > if the fitness is not 1, repeat the loop. Seems like function otimization problem.
"""

while(True):
    print("Generation: ", i)
    initialPopulation = clf.sortByFitness(initialPopulation)
    newPopulation = []
    for i in range(5):
        newPopulation.append(initialPopulation[i])
        newPopulation.append(initialPopulation[i])
    for i in range(5):
        newPopulation.append(clf.mutate(initialPopulation[i]))
    initialPopulation = newPopulation
    if clf.checkIfFitnessDoesntChange(clf.calculateFitness(x_train, x_test, y_train, y_test, *initialPopulation[0])):
        print("Best individual: ", initialPopulation[0])
        print("Fitness: ", clf.calculateFitness(
            x_train, x_test, y_train, y_test, *initialPopulation[0]))
        break
    print("Best individual: ", initialPopulation[0])
    print("Fitness: ", clf.calculateFitness(
        x_train, x_test, y_train, y_test, *initialPopulation[0]))
    i += 1
