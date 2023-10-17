import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def c45_tree(df, target):

    # Cria o classificador DecisionTreeClassifier
    clf = DecisionTreeClassifier(criterion="entropy", splitter="best")

    # Treina o classificador
    clf.fit(df, target)

    return clf


data = pd.read_csv("treino_sinais_vitais_com_label.txt")
data = data.drop(data.columns[[0, 1, 2]], axis = 1)

# print(data.to_string(index = False))

x = data.iloc[:, [0, 1, 2, 3]]
y = data.iloc[:, 4]

# Cria o classificador
clf = DecisionTreeClassifier(criterion="entropy", splitter="best")

# Treina o classificador
clf.fit(x, y)
