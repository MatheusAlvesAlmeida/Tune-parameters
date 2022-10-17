import random
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

criterions = ['gini', 'entropy']
splits = ['best', 'random']
lower_depth, upper_depth = 2, 6
lower_leaf_nodes, upper_leaf_nodes = 2, 6
class_weight = ['balanced', None]
lower_samples_leaf, upper_samples_leaf = 0.01, 0.5

last10 = []


def getTrainTestVariables(df):
    return train_test_split(df.drop('Survived', axis=1), df['Survived'], test_size=0.30)


def calculateFitness(X_train, X_test, y_train, y_test, criterion, splitter, max_depth, max_leaf_nodes, class_weight, min_samples_leaf):
    model = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth,
                                   min_samples_leaf=min_samples_leaf, class_weight=class_weight, max_leaf_nodes=max_leaf_nodes)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return classification_report(y_test, predictions, output_dict=True)['accuracy']


def generatePopupation(size):
    params = []
    for i in range(size):
        params.append([random.choice(criterions), random.choice(splits), random.randint(lower_depth, upper_depth),
                       random.randint(lower_leaf_nodes, upper_leaf_nodes), random.choice(
            class_weight),
            float("{0:.3f}".format(
                random.uniform(lower_samples_leaf, upper_samples_leaf)))])

    return params


def sortByFitness(population, x_train, x_test, y_train, y_test):
    return sorted(population, key=lambda x: calculateFitness(x_train, x_test, y_train, y_test, *x), reverse=True)


def checkIfFitnessDoesntChange(newFitness):
    if len(last10) == 15:
        if all(x == last10[0] for x in last10):
            return True
        last10.pop(0)
        last10.append(newFitness)
        return False
    last10.append(newFitness)
    return False


def mutate(individual):
    gene = random.randint(0, 6)
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
    elif gene == 3:
        individual[3] = random.randint(lower_leaf_nodes, upper_leaf_nodes)
    elif gene == 4:
        if individual[4] == 'balanced':
            individual[4] = None
        else:
            individual[4] = 'balanced'
    elif gene == 5:
        individual[5] = float("{0:.3f}".format(
            random.uniform(lower_samples_leaf, upper_samples_leaf)))

    return individual

# Whole arithmatic crossover


def crossover(parent1, parent2, alpha=0.5):
    size = 6
    child = [None] * size
    # Add the two first genes
    child[0] = parent1[0]
    child[1] = parent1[1]
    for i in [2, 3, 5]:
        child[i] = (alpha * parent1[i] + (1 - alpha) * parent2[i])
    child[4] = parent1[4]
    # cast index 2 and 3 to int
    child[2] = int(child[2])
    child[3] = int(child[3])
    return child


def parentSelection(population, x_train, x_test, y_train, y_test):
    parents = []
    for i in range(10):
        parents.append(random.choice(population))
    parents = sortByFitness(parents, x_train, x_test, y_train, y_test)
    return [parents[0], parents[1]]
