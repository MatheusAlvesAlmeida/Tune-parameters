import pandas as pd
from prepare_data import getTratedData
import classifier as clf

df = getTratedData()

x_train, x_test, y_train, y_test = clf.getTrainTestVariables(df)

initialPopulation = clf.generatePopupation(10)

i = 1

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
