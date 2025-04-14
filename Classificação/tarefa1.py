# Objetivo: treinar um modelo de classificação para prever se um celular possui bateria alta ou baixa com base em suas características técnicas.
# Escolha do algoritmo:  Random Forest, pois ele lida bem com dados numéricos e categóricos sem exigir muitas transformações.

# Para rodar:
# py -m venv venv 
# pip install pandas
# pip install scikit-learn
# python .\tarefa1.pyg

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

print("Carregando os dados...")
df = pd.read_csv("test.csv")
print("Dados carregados com sucesso!\n")

print("Visualizando as primeiras linhas do dataset:")
print(df.head(), "\n")

print("Removendo colunas identificadoras...")
df.drop(columns=["id"], inplace=True)
print("Coluna 'id' removida!\n")

print("Verificando valores nulos antes da remoção:")
print(df.isnull().sum(), "\n")

df.dropna(inplace=True)

print("Valores nulos removidos!\n")
print("Verificando valores nulos após a remoção:")
print(df.isnull().sum(), "\n")

# Criar a variável-alvo: classificar bateria como Alta (1) ou Baixa (0) baseado na mediana
# Ela define o que se pretende prever ou estimar e serve como base para a construção do modelo.
# Se a variavel-alvo battery_power for maior que a mediana a bateria será classificada como alta e se for menor que a mediana será baixa.

print("Criando variável-alvo para classificação da bateria...")
median_battery = df["battery_power"].median()
df["high_battery"] = (df["battery_power"] > median_battery).astype(int)
print("Variável-alvo criada com sucesso!\n")
print("Visualizando as primeiras linhas do dataset:")
print(df.head(), "\n")

# Remover a coluna 'battery_power' para evitar vazamento de dados
print("Removendo a coluna 'battery_power'...")
df.drop(columns=["battery_power"], inplace=True)
print("Coluna 'battery_power' removida!\n")
print("Visualizando as primeiras linhas do dataset:")
print(df.head(), "\n")

# Separar variáveis independentes e dependentes
print("Separando variáveis independentes (X) e dependentes (y)...")
X = df.drop(columns=["high_battery"])
y = df["high_battery"]
print("Separação concluída!\n")

# Padronizar colunas numéricas
print("Padronizando os dados...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Padronização concluída!\n")

# Dividir os dados em treino e teste
print("Dividindo os dados em treino e teste...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
print(f"Total de amostras de treino: {X_train.shape[0]}")
print(f"Total de amostras de teste: {X_test.shape[0]}")
print("Divisão concluída!\n")

# Criar e treinar o modelo
print("Treinando o modelo Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Modelo treinado com sucesso!\n")

# Fazer previsões
y_pred = model.predict(X_test)

# Avaliação do modelo
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


# Novo celular para prever a bateria alta ou baixa
novo_celular = {
    'blue': 1,              # Bluetooth (1 ou 0)
    'clock_speed': 2.2,     # Velocidade do processador (GHz)
    'dual_sim': 1,          # Dual SIM (1 ou 0)
    'fc': 8,                # Câmera frontal (megapixels)
    'four_g': 1,            # 4G (1 ou 0)
    'int_memory': 64,       # Memória interna (GB)
    'm_dep': 0.8,           # Profundidade do celular (cm)
    'mobile_wt': 150,       # Peso do celular (gramas)
    'n_cores': 8,           # Número de núcleos do processador
    'pc': 4,                # Memória RAM (GB)
    'px_height': 1920,      # Altura da tela (pixels)
    'px_width': 1080,       # Largura da tela (pixels)
    'ram': 4096,            # Quantidade de RAM (em MB)
    'sc_h': 6.1,            # Altura da tela (em cm)
    'sc_w': 3.4,            # Largura da tela (em cm)
    'talk_time': 10,        # Tempo de conversação (horas)
    'three_g': 1,           # 3G (1 ou 0)
    'touch_screen': 1,      # Tela sensível ao toque (1 ou 0)
    'wifi': 1               # Wi-Fi (1 ou 0)
}

# Convertendo para DataFrame com uma única linha
novo_celular_df = pd.DataFrame([novo_celular])

# Exibindo o novo celular
print("Novo celular:", novo_celular_df)

# Garantindo que as colunas do novo celular correspondam às usadas no treinamento
novo_celular_df = novo_celular_df[X.columns]  # Alinha as colunas para corresponder

# Padronizando os dados do novo celular com o mesmo scaler usado no treino
novo_celular_scaled = scaler.transform(novo_celular_df)

# Prevendo a classe (bateria alta ou baixa) com o modelo treinado
previsao = model.predict(novo_celular_scaled)

# Exibindo o resultado da previsão
if previsao[0] == 1:
    print("A previsão é: Bateria Alta")
else:
    print("A previsão é: Bateria Baixa")
