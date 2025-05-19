### Objetivo: realizar uma análise de agrupamento (clustering) nos dados de imóveis para identificar padrões de comportamento ou perfis semelhantes entre as casas,

### Tratamento de dados:
### One-Hot Encoding: para converter dados categóricos em formato numérico.
### Normalização (StandardScaler): para que todas as variáveis tenham a mesma importância.
### PCA (Análise de Componentes Principais): para reduzir a dimensionalidade dos dados e visualizar os clusters em 2D e 3D.

### Escolha do número de clusters: Método cotovelo
### Para cada valor de k, calcula-se o WSS (Within-cluster Sum of Squares)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.cm as cm

# Leitura dos dados
df = pd.read_csv('test.csv')

# One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=['Location', 'Condition', 'Garage'])

# Remover colunas não usadas na clusterização
X = df_encoded.drop(['Id', 'Price'], axis=1)

# Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Escolha do número de clusters pelo método do cotovelo
wss = []
k_range = range(1, 15)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, wss, marker='o')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('WSS (Within-cluster Sum of Squares)')
plt.title('Método do Cotovelo')
plt.grid(True)
plt.show()

# Aplicar KMeans com número definido de clusters
num_clusters = 14
model = KMeans(n_clusters=num_clusters, random_state=42)
model.fit(X_scaled)
group = model.predict(X_scaled)

# PCA com 2 componentes para visualização 2D
pca_2d = PCA(n_components=2)
pca_array_2d = pca_2d.fit_transform(X_scaled)

df_pca_2d = pd.DataFrame(data=pca_array_2d, columns=['PC1', 'PC2'])
df_pca_2d['group'] = group

colors = [cm.tab20(i / num_clusters) for i in range(num_clusters)]
df_pca_2d['color'] = df_pca_2d['group'].map({i: colors[i] for i in range(num_clusters)})
df_pca_2d['color'] = df_pca_2d['color'].fillna('#000000')

plt.figure(figsize=(8, 6))
plt.scatter(df_pca_2d['PC1'], df_pca_2d['PC2'], c=df_pca_2d['color'], alpha=0.6, s=10)
plt.title("Clusters de Imóveis (PCA 2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()

# PCA com 3 componentes para visualização 3D
pca_3d = PCA(n_components=3)
pca_array_3d = pca_3d.fit_transform(X_scaled)

df_pca_3d = pd.DataFrame(data=pca_array_3d, columns=['PC1', 'PC2', 'PC3'])
df_pca_3d['group'] = group
df_pca_3d['color'] = df_pca_3d['group'].map({i: colors[i] for i in range(num_clusters)})
df_pca_3d['color'] = df_pca_3d['color'].fillna('#000000')

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_pca_3d['PC1'], df_pca_3d['PC2'], df_pca_3d['PC3'], c=df_pca_3d['color'], s=20, alpha=0.7)
ax.set_title("Clusters de Imóveis (PCA 3D)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
plt.show()

# Adiciona o número do cluster ao DataFrame original
df['cluster'] = group

# Calcula a média de preço por cluster
mean_price_per_cluster = df.groupby('cluster')['Price'].mean().reset_index()

# Renomeia a coluna para clareza
mean_price_per_cluster.columns = ['Cluster', 'Preço Médio']

# Exibe a tabela ordenada por preço médio
mean_price_per_cluster = mean_price_per_cluster.sort_values(by='Preço Médio', ascending=False)
print(mean_price_per_cluster)

# === VISUALIZAÇÃO 1: Gráfico de Barras ===
plt.figure(figsize=(10, 6))
plt.bar(mean_price_per_cluster['Cluster'], mean_price_per_cluster['Preço Médio'], color='skyblue')
plt.title('Preço Médio por Cluster')
plt.xlabel('Cluster')
plt.ylabel('Preço Médio')
plt.xticks(mean_price_per_cluster['Cluster'])
plt.grid(axis='y')
plt.show()

# Agrupa por cluster e calcula a média de todas as colunas numéricas
mean_attributes_per_cluster = df.groupby('cluster').mean(numeric_only=True).reset_index()

# Seleciona colunas principais para visualização
cols_to_show = ['cluster', 'Bedrooms', 'Bathrooms', 'Area', 'Price']
print(mean_attributes_per_cluster[cols_to_show].sort_values(by='Price', ascending=False))