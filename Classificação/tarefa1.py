# Objetivo: treinar um modelo de classificação para prever se um celular possui bateria alta ou baixa com base em suas características técnicas.
# Escolha do algoritmo:  Random Forest, pois ele lida bem com dados numéricos e categóricos sem exigir muitas transformações.

# Para rodar:
# py -m venv venv 
# pip install pandas
# pip install scikit-learn
# python .\tarefa1.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# 1. Carregar os dados
print("Carregando os dados...")
df = pd.read_csv("test.csv")
print("Dados carregados com sucesso!\n")

# 2. Exibir as primeiras linhas do dataset
print("Visualizando as primeiras linhas do dataset:")
print(df.head(), "\n")

# 3. Remover colunas identificadoras (exemplo: id)
print("Removendo colunas identificadoras...")
df.drop(columns=["id"], inplace=True)
print("Coluna 'id' removida!\n")

# 4. Remover valores nulos
print("Verificando valores nulos antes da remoção:")
print(df.isnull().sum(), "\n")

df.dropna(inplace=True)

print("Valores nulos removidos!\n")
print("Verificando valores nulos após a remoção:")
print(df.isnull().sum(), "\n")

# 5. Criar a variável-alvo: classificar bateria como Alta (1) ou Baixa (0) baseado na mediana
print("Criando variável-alvo para classificação da bateria...")
median_battery = df["battery_power"].median()
df["high_battery"] = (df["battery_power"] > median_battery).astype(int)
print("Variável-alvo criada com sucesso!\n")

# 6. Remover a coluna original 'battery_power' para evitar vazamento de dados
print("Removendo a coluna 'battery_power'...")
df.drop(columns=["battery_power"], inplace=True)
print("Coluna 'battery_power' removida!\n")

# 7. Separar variáveis independentes e dependentes
print("Separando variáveis independentes (X) e dependentes (y)...")
X = df.drop(columns=["high_battery"])
y = df["high_battery"]
print("Separação concluída!\n")

# 8. Padronizar colunas numéricas
print("Padronizando os dados...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Padronização concluída!\n")

# 9. Dividir os dados em treino e teste
print("Dividindo os dados em treino e teste...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
print(f"Total de amostras de treino: {X_train.shape[0]}")
print(f"Total de amostras de teste: {X_test.shape[0]}")
print("Divisão concluída!\n")

# 10. Criar e treinar o modelo
print("Treinando o modelo Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Modelo treinado com sucesso!\n")

# 11. Fazer previsões
y_pred = model.predict(X_test)

# 12. Avaliação do modelo
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nMétricas de Avaliação do Modelo:")
print(f"Acurácia: {accuracy:.2f}")
print(f"Precisão: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}\n")

# Exibir relatório completo de classificação
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))
