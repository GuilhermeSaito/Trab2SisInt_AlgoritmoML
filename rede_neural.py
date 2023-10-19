import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import mean_squared_error, accuracy_score

# Usando regressao
def train_continuos_values(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print(f'Erro Médio Quadrático (MSE): {mse:.2f}')

# Usando classificacao
def train_discret_values(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Acurácia do modelo: {accuracy:.2f}')


nomes_colhnas_com_label = ["qPA", "pulso", "freq", "gravidade", "classes"]
data = pd.read_csv("treino_sinais_vitais_com_label.txt", header = None, names = nomes_colhnas_com_label)

x = data.loc[:, "qPA" : "freq"]
y = data.loc[:, "gravidade"]
y_classe = data.loc[:, "classes"]

x = x.reset_index(drop=True)
y = y.reset_index(drop=True)
y_classe = y_classe.reset_index(drop=True)

train_continuos_values(x, y)
train_discret_values(x, y_classe.values)