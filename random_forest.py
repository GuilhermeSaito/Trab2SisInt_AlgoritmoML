import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt

# Usando regressao
def train_continuos_values(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print(f'Erro Médio Quadrático (MSE): {mse:.2f}')

# Usando classificacao
def train_discret_values(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(
        n_estimators = 1000,
        # max_depth = 14,                       # Como sao somente 3 colunas para treinamento 1 de teste, acredito q a profundidade da arvore nao va afetar
        max_leaf_nodes = 200,
        bootstrap = False,
        n_jobs = 16,
        random_state = 1,
        verbose = True
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Acurácia do modelo: {accuracy:.2f}')

    feature_importances = model.feature_importances_
    feature_names = range(len(feature_importances))
    # print(model.feature_names_in_)
    
    plt.bar(model.feature_names_in_, feature_importances)
    plt.xlabel('Características')
    plt.ylabel('Importância')
    plt.title('Importância de Características')
    plt.grid(True)
    # plt.legend()
    # plt.show()
    plt.savefig('validacao_loss_regressao.png')

nomes_colhnas_com_label = ["qPA", "pulso", "freq", "gravidade", "classes"]
data = pd.read_csv("treino_sinais_vitais_com_label.txt", header = None, names = nomes_colhnas_com_label)

x = data.loc[:, "qPA" : "freq"]
y = data.loc[:, "gravidade"]
y_classe = data.loc[:, "classes"]

x = x.reset_index(drop=True)
y = y.reset_index(drop=True)
y_classe = y_classe.reset_index(drop=True)

# train_continuos_values(x, y)
train_discret_values(x, y_classe)