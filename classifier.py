import random
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

criterions = ['gini', 'entropy']
splits = ['best', 'random']
lower_depth, upper_depth = 2, 100
lower_leaf_nodes, upper_leaf_nodes = 2, 100
class_weight = ['balanced', None]
lower_samples_leaf, upper_samples_leaf = 0, 0.5

last10 = []


def getTrainTestVariables(df):
    return train_test_split(df.drop('Survived', axis=1), df['Survived'], test_size=0.30)


def calculateFitness(X_train, X_test, y_train, y_test, criterion, splitter, max_depth, min_samples_leaf, class_weight):
    model = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth,
                                   min_samples_leaf=min_samples_leaf, class_weight=class_weight)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return classification_report(y_test, predictions, output_dict=True)['accuracy']


def generatePopupation(size):
    params = []
    for i in range(size):
        params.append([random.choice(criterions), random.choice(splits), random.randint(lower_depth, upper_depth),
                       random.randint(lower_leaf_nodes, upper_leaf_nodes), random.choice(
            class_weight),
            random.uniform(lower_samples_leaf, upper_samples_leaf)])

    return params


def sortByFitness(population):
    return sorted(population, key=calculateFitness(), reverse=True)


def checkIfFitnessDoesntChange(newFitness):
    if len(last10) == 10:
        last10.pop(0)
        last10.append(newFitness)
        return False
    if len(last10) < 10:
        last10.append(newFitness)
        return False
    return all(element == last10[0] for element in last10)


def mutate(individual):
    gene = random.randint(0, 9)
    if gene == 0:
        if individual[0] == 'gini':
            individual[0] = 'entropy'
        else:
            individual[0] = 'gini'
    elif gene == 1:
        if individual[1] == 'best':
            individual[1] = 'random'
        else:
            individual[1] = 'best'
    elif gene == 2:
        individual[2] = random.randint(lower_depth, upper_depth)
    elif gene in [4, 5]:
        individual[gene] = random.uniform(
            lower_samples_leaf, upper_samples_leaf)
    elif gene in [3, 6, 8]:
        individual[gene] = random.random()
    elif gene == 7:
        individual[7] = random.randint(lower_leaf_nodes, upper_leaf_nodes)
    elif gene == 9:
        if individual[9] == 'balanced':
            individual[9] = None
        else:
            individual[9] = 'balanced'

    return individual


def parentSelection(population):
    parents = []
    for i in range(10):
        parents.append(random.choice(population))
    parents = sortByFitness(parents)
    return [parents[0], parents[1]]
