import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("treino_sinais_vitais_com_label.txt")
data = data.drop(data.columns[[0, 1, 2]], axis = 1)

# print(data.to_string(index = False))

x = data.iloc[:, [0, 1, 2, 3]]
y = data.iloc[:, 4]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo: {accuracy:.2f}')