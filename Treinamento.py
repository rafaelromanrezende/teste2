import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Carregar os dados
dados = pd.read_csv('dados_gestos.csv')

# Separar rótulos e coordenadas
X = dados.iloc[:, 1:].values  # Dados dos landmarks
y = dados.iloc[:, 0].values   # Rótulos dos gestos

# Dividir os dados em treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar o modelo
modelo = RandomForestClassifier(n_estimators=100)
modelo.fit(X_treino, y_treino)

# Avaliar o modelo
precisao = modelo.score(X_teste, y_teste)
print(f'Acurácia do modelo: {precisao:.2f}')

# Salvar o modelo treinado
joblib.dump(modelo, 'modelo_gestos.pkl')
