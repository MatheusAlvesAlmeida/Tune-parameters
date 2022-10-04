import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preencherIdades(cols):
    idade = cols[0]
    pClass = cols[1]
    if pd.isnull(idade):
        if(pClass == 1):
            return 37
        elif (pClass == 2):
            return 29
        else:
            return 24
    else:
        return idade


def dropColumns(df):
    df['Age'] = df[['Age', 'Pclass']].apply(preencherIdades, axis=1)
    df.drop('Cabin', axis=1, inplace=True)
    df.dropna(inplace=True)
    return df


def finalTreatment(df):
    label = LabelEncoder()
    for col in ['Sex', 'Embarked']:
        df[col] = label.fit_transform(df[col])
    df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
    return df


def getTratedData():
    df = pd.read_csv('./Data/titanic_train.csv')
    df = dropColumns(df)
    df = finalTreatment(df)
    return df
