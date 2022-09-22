from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def getTrainTestVariables(df):
    return train_test_split(df.drop('Survived', axis=1), df['Survived'], test_size=0.30)


def getPredictions(X_train, X_test, y_train, y_test):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return classification_report(y_test, predictions, output_dict=True)
