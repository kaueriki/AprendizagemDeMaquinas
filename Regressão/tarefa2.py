# Regressão a variavel alvo é de categoria numérica
# métricas de avaliação: Mean Squared Error e Root Mean Squared Error

#base de dados : time series
# Peguei uma base de dados com valores sazonal de vendas de caminhões
# Serie temporal:
# Prever valore futuros
# Preparação de dados: Cada registro contém uma certa quanridade de valores anteriores
# Variaveis exogenas: Possuem valores que afetam a saida, mas não são afetadas por outras váriaveis de entrada, já estão disponíveis os valores futuros
# Valor de lag: Vai depende do tamaho da temporada, se não tem repetição tem que ficar testando.
# https://dontpad/aprendizagem_maquina
# não usar skforecast para, só se for estacionario
# Foi utilizado o modelo Arima Autoregressivo integrated Média average

# Objetivo: Prever vendas de caminhões ao longo do tempo
# Tratamento de dados: A coluna "Month-Year" foi convertida para o tipo datetime, para servir como index e garantir que a série fique em ordem cronológica
# A base era limpa, então não houve necessidade de padronizar ou prrencher colunas de dados ausentes
# Algoritmo: Foi utilizado Sarima pois a serie não é estacionária e sim sazonal, e tem apenas uma variável externa

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")
import mean_squared_error

# 1. Carregar os dados
print("Carregando os dados...")
df = pd.read_csv("Truck_sales.csv")
print("Dados carregados com sucesso!\n")

# 2. Converter coluna de data e ordenar
print("Convertendo coluna de data...")
df["Month-Year"] = pd.to_datetime(df["Month-Year"], format="%y-%b")
df = df.sort_values("Month-Year")
df.set_index("Month-Year", inplace=True)
print("Conversão concluída!\n")

# 3. Plot da série temporal
plt.figure(figsize=(10, 4))
df["Number_Trucks_Sold"].plot(title="Vendas de Caminhões ao Longo do Tempo")
plt.xlabel("Ano")
plt.ylabel("Caminhões Vendidos")
plt.tight_layout()
plt.show()

# 4. Separar treino e teste (últimos 12 meses)
train = df.iloc[:-12]
test = df.iloc[-12:]

# 5. Criar e treinar modelo ARIMA com sazonalidade de 12 meses
print("Treinando modelo ARIMA com sazonalidade...\n")
model = auto_arima(train["Number_Trucks_Sold"],
                   seasonal=True,
                   m=12, 
                   trace=True,
                   error_action='ignore',
                   suppress_warnings=True,
                   stepwise=True)

model.fit(train["Number_Trucks_Sold"])
print("\nModelo treinado com sucesso!\n")

# 6. Fazer previsões
forecast = model.predict(n_periods=12)
forecast = pd.DataFrame(forecast, index=test.index, columns=["Prediction"])

# 7. Avaliação do modelo

print("\nMétricas de Avaliação do Modelo:")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}\n")

plt.figure(figsize=(10, 4))
plt.plot(train.index, train["Number_Trucks_Sold"], label="Treino")
plt.plot(test.index, test["Number_Trucks_Sold"], label="Real")
plt.plot(test.index, forecast["Prediction"], label="Previsto (ARIMA sazonal)", linestyle="--")
plt.title("Previsão de Vendas de Caminhões (ARIMA sazonal)")
plt.xlabel("Ano")
plt.ylabel("Nº de Caminhões Vendidos")
plt.legend()
plt.tight_layout()
plt.show()