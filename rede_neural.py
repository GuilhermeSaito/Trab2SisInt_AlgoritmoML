import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt

# Usando regressao
def train_continuos_values(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    quantidade_neuronio_camada = []
    quantidade_camada = 10
    for i in range(quantidade_camada):
        quantidade_neuronio_camada.append(100)

    model = MLPRegressor(
        hidden_layer_sizes = quantidade_neuronio_camada,     # 1 camada oculta com somente 100 neuronio
        activation = "relu",                                 # returns f(x) = max(0, x)
        solver = "adam",                                     # adam = SGD mas com alteracoes de uns manos lah.
        learning_rate = "constant",         
        learning_rate_init = 0.01,                           # ------------- Nesse modelo, altera esse parametro aqui para ver a diferenca no erro quadratico emdio
        max_iter = 1000,
        random_state = 1,                                    # Para deixar a "aleatoriedade" consistente, para conseguir verificar o impacto dos hiperparametros, reproduzir esse mesmo teste diferentes vezes, etc. Nao mexer aqui
        verbose = True,
        # alpha = 1,
        early_stopping=True,                                 
        n_iter_no_change = 100,                             # Parametro para ver quantas epocas o loss nao baixa por um numero x ou a otimizacao nao melhora
        shuffle = True,
        momentum = 1,
        nesterovs_momentum = True
        )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # O erro quadratico medio = Media da diferenca de cada valor respectivivo em modulo ou seja
    # x[0] = sqrt(abs(pow(a, 2) - pow(b, 2))) ...
    # mean(y[0], y[1], ...)
    # Entao, quanto menor esse erro, mais o modelo estah acertando os valores previstos
    mse = mean_squared_error(y_test, y_pred)
    print(f'Erro Médio Quadrático (MSE): {mse:.2f}')

    loss_curve = model.loss_curve_
    validation_curve = model.validation_scores_

    plt.plot(range(1, len(loss_curve) + 1), loss_curve, label='Curva Loss')
    plt.plot(range(1, len(validation_curve) + 1), validation_curve, label='Pontuação de Validação')
    plt.title('Curva de Validação')
    plt.xlabel('Épocas')
    plt.ylabel('Pontuação')
    plt.grid(True)
    plt.legend()
    # plt.show()
    plt.savefig('validacao_loss_regressao.png')

# Usando classificacao
def train_discret_values(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    quantidade_neuronio_camada = []
    quantidade_camada = 10
    for i in range(quantidade_camada):
        quantidade_neuronio_camada.append(100)

    model = MLPClassifier(
        hidden_layer_sizes = quantidade_neuronio_camada,     # 1 camada oculta com somente 100 neuronio
        activation = "relu",                                 # returns f(x) = max(0, x)
        solver = "sgd",                                      # SGD = Stochastic Gradient Descent.
        learning_rate = "constant",         
        learning_rate_init = 0.00001,            
        max_iter = 1000,
        random_state = 1,                                    # Para deixar a "aleatoriedade" consistente, para conseguir verificar o impacto dos hiperparametros, reproduzir esse mesmo teste diferentes vezes, etc. Nao mexer aqui
        verbose = True,
        # alpha = 1,
        early_stopping=True,                                 
        n_iter_no_change = 1000,                             # Parametro para ver quantas epocas o loss nao baixa por um numero x ou a otimizacao nao melhora
        shuffle = True,
        momentum = 1,
        nesterovs_momentum = True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Acurácia do modelo: {accuracy:.2f}')

    loss_curve = model.loss_curve_
    validation_curve = model.validation_scores_

    plt.plot(range(1, len(loss_curve) + 1), loss_curve, label='Curva Loss')
    plt.plot(range(1, len(validation_curve) + 1), validation_curve, label='Pontuação de Validação')
    plt.title('Curva de Validação')
    plt.xlabel('Épocas')
    plt.ylabel('Pontuação')
    plt.grid(True)
    plt.legend()
    # plt.show()
    plt.savefig('validacao_loss.png')


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