import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt

def train_continuos_values_hiperparametros_otimizados(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    param_grid = {
        "n_estimators": [1, 10, 100, 1000],
        "criterion": ["squared_error", "absolute_error", "friedman_mse", "poisson"],
        "max_depth": [1, 4, 8, 10, 14, 20],
        "max_leaf_nodes": [1, 10, 50, 100, 150, 200],
        "bootstrap": [True, False]
    }

    model = RandomForestRegressor(random_state = 1)
    grid_search = GridSearchCV(model, param_grid, cv = 5, scoring = 'neg_mean_squared_error', verbose = 2)

    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    print("Melhores hiperparametros Random Forest Continuo (Gravidade):", best_params)

    melhor_model = RandomForestRegressor(random_state = 1, **best_params)
    melhor_model.fit(X_train, y_train)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print(f'Erro Médio Quadrático (MSE): {mse:.2f}')

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
    plt.savefig('random_forest_gravidade_valoresAutomatizados.png')
    plt.clf()

# Usando classificacao
def train_discret_values_hiperparametros_otimizados(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    param_grid = {
        "n_estimators": [1, 10, 100, 1000],
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": [1, 4, 8, 10, 14, 20],
        "max_leaf_nodes": [1, 10, 50, 100, 150, 200],
        "bootstrap": [True, False]
    }

    model = RandomForestClassifier(random_state = 1)
    grid_search = GridSearchCV(model, param_grid, cv = 5, scoring = 'neg_mean_squared_error', verbose = 2)

    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    print("Melhores hiperparametros Random Forest Discreto (Classe da gravidade):", best_params)

    melhor_model = RandomForestClassifier(random_state = 1, **best_params)
    melhor_model.fit(X_train, y_train)

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
    plt.savefig('random_forest_classificacaoGravidade_valoresAutomatizados.png')
    plt.clf()

# Usando regressao
def train_continuos_values(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators = 1000,
        criterion = "poisson",
        max_depth = 14,                       # Como sao somente 3 colunas para treinamento 1 de teste, acredito q a profundidade da arvore nao va afetar
        max_leaf_nodes = 200,
        bootstrap = True,
        n_jobs = 16,
        random_state = 1,
        verbose = True
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print(f'Erro Médio Quadrático (MSE): {mse:.2f}')

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
    plt.savefig('random_forest_gravidade_valoresOtimizados.png')
    plt.clf()

# Usando classificacao
def train_discret_values(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(
        n_estimators = 1000,
        criterion = "entropy",
        max_depth = 14,                       # Como sao somente 3 colunas para treinamento 1 de teste, acredito q a profundidade da arvore nao va afetar
        max_leaf_nodes = 200,
        bootstrap = True,
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
    plt.savefig('random_forest_classificacaoGravidade_valoresOtimizados.png')
    plt.clf()

nomes_colhnas_com_label = ["qPA", "pulso", "freq", "gravidade", "classes"]
data = pd.read_csv("treino_sinais_vitais_com_label.txt", header = None, names = nomes_colhnas_com_label)

x = data.loc[:, "qPA" : "freq"]
y = data.loc[:, "gravidade"]
y_classe = data.loc[:, "classes"]

x = x.reset_index(drop=True)
y = y.reset_index(drop=True)
y_classe = y_classe.reset_index(drop=True)

# train_continuos_values(x, y)
# train_discret_values(x, y_classe)
# train_continuos_values_hiperparametros_otimizados(x, y)
train_discret_values_hiperparametros_otimizados(x, y_classe)