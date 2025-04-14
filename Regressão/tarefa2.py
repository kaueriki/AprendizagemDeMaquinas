# Regressão a variavel alvo é de categoria numérica
# métricas de avaliação: Mean Squared Error e Root Mean Squared Error

#base de dados : time series
# Peguei uma base de dados com valores de vendas de caminhões

# Serie temporal:
# Prever valore futuros
# Preparação de dados: Cada registro contém uma certa quanridade de valores anteriores
# Variaveis exogenas: Possuem valores que afetam a saida, mas não são afetadas por outras váriaveis de entrada, já estão disponíveis os valores futuros
# Valor de lag: Vai depende do tamaho da temporada, se não tem repetição tem que ficar testando.
# https://dontpad/aprendizagem_maquina
# não usar skforecast para, só se for estacionario

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# 1. Carregar os dados
print("Carregando os dados...")
df = pd.read_csv("Truck_sales.csv")
print("Dados carregados com sucesso!\n")

# 2. Visualizar os dados iniciais
print("Visualizando os dados:")
print(df.head(), "\n")

# 3. Converter coluna de data e ordenar
print("Convertendo coluna de data...")
df["Month-Year"] = pd.to_datetime(df["Month-Year"], format="%y-%b")
df = df.sort_values("Month-Year")
df.set_index("Month-Year", inplace=True)
print("Conversão concluída!\n")

# 4. Exibir gráfico da série temporal
plt.figure(figsize=(10, 4))
df["Number_Trucks_Sold"].plot(title="Vendas de Caminhões ao Longo do Tempo")
plt.xlabel("Ano")
plt.ylabel("Caminhões Vendidos")
plt.tight_layout()
plt.show()

# 5. Separar treino e teste (ex: últimos 12 meses para teste)
train = df.iloc[:-12]
test = df.iloc[-12:]

# 6. Treinar modelo ARIMA
print("Treinando modelo ARIMA...")
model = ARIMA(train, order=(2,1,2))
model_fit = model.fit()
print("Modelo treinado com sucesso!\n")

# 7. Fazer previsões
print("Fazendo previsões para os 12 meses finais...")
pred = model_fit.forecast(steps=12)

# 8. Avaliação do modelo
mse = mean_squared_error(test, pred)
mae = mean_absolute_error(test, pred)
rmse = np.sqrt(mse)

print("\nMétricas de Avaliação do Modelo:")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}\n")

# 9. Plotar previsões x valores reais
plt.figure(figsize=(10, 4))
plt.plot(train.index, train["Number_Trucks_Sold"], label="Treino")
plt.plot(test.index, test["Number_Trucks_Sold"], label="Real")
plt.plot(test.index, pred, label="Previsto", linestyle="--")
plt.title("Previsão de Vendas de Caminhões (ARIMA)")
plt.xlabel("Ano")
plt.ylabel("Nº de Caminhões Vendidos")
plt.legend()
plt.tight_layout()
plt.show()