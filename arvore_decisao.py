import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def train_values(x, y, cont):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier(
        criterion = "entropy",
        splitter = "best",
        max_depth = 14,
        random_state = 1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Acurácia do modelo: {accuracy:.2f}')

    feature_importances = model.feature_importances_
    # feature_names = range(len(feature_importances))
    print(model.feature_names_in_)
    
    plt.bar(model.feature_names_in_, feature_importances)
    plt.xlabel('Características')
    plt.ylabel('Importância')
    plt.title('Importância de Características')
    plt.grid(True)
    # plt.legend()
    # plt.show()
    plt.savefig('arvore_decisao_gravidade_classeGravidade_valoresOtimizados' + str(cont) + '.png')
    plt.clf()

nomes_colhnas_com_label = ["qPA", "pulso", "freq", "gravidade", "classes"]
data = pd.read_csv("treino_sinais_vitais_com_label.txt", header = None, names = nomes_colhnas_com_label)

x = data.loc[:, "qPA" : "freq"]
y = data.loc[:, "gravidade"]
y_classe = data.loc[:, "classes"]

y_discretizado = pd.cut(y, 10, labels=False)

x = x.reset_index(drop=True)
y = y.reset_index(drop=True)
y_discretizado = y_classe.reset_index(drop=True)

train_values(x, y_classe, 0)
train_values(x, y_discretizado, 1)