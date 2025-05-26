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
from math import sqrt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

def optimal_number_of_clusters(wcss):
    x1, y1 = 2, wcss[0]
    x2, y2 = 20, wcss[-1]

    distances = []
    for i in range(len(wcss)):
        x0 = i + 2  # começa em 2
        y0 = wcss[i]
        numerator = abs((y2 - y1)*x0 - (x2 - x1)*y0 + x2*y1 - y2*x1)
        denominator = sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator / denominator)
    
    return distances.index(max(distances)) + 2

# Leitura dos dados
df = pd.read_csv('test.csv')

# One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=['Location', 'Condition', 'Garage'])

# Remover colunas não usadas na clusterização
X = df_encoded.drop(['Id', 'Price'], axis=1)

# Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cálculo do WCSS para vários valores de k
wcss = []
k_range = range(2, 21)  # de 2 a 20 clusters
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Determinar número ideal de clusters
optimal_k = optimal_number_of_clusters(wcss)
print(f"Melhor número de clusters (Método do Cotovelo Automático): {optimal_k}")

plt.figure(figsize=(8, 5))
plt.plot(k_range, wcss, marker='o')
plt.axvline(x=optimal_k, color='red', linestyle='--', label=f'k ótimo = {optimal_k}')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('WSS (Within-cluster Sum of Squares)')
plt.title('Método do Cotovelo Automático')
plt.legend()
plt.grid(True)
plt.show()

# Aplicar KMeans com número ótimo de clusters
num_clusters = optimal_k
model = KMeans(n_clusters=num_clusters, random_state=42)
model.fit(X_scaled)
group = model.predict(X_scaled)

centroids = model.cluster_centers_
centroids_df = pd.DataFrame(centroids, columns=X.columns)
centroids_df['cluster'] = range(num_clusters)

centroids_original = scaler.inverse_transform(centroids)
centroids_original_df = pd.DataFrame(centroids_original, columns=X.columns)
centroids_original_df['cluster'] = range(num_clusters)

# PCA 2D
pca_2d = PCA(n_components=2)
pca_array_2d = pca_2d.fit_transform(X_scaled)
centroids_pca_2d = pca_2d.transform(centroids)

df_pca_2d = pd.DataFrame(data=pca_array_2d, columns=['PC1', 'PC2'])
df_pca_2d['group'] = group

colors = [cm.tab20(i / num_clusters) for i in range(num_clusters)]
df_pca_2d['color'] = df_pca_2d['group'].map({i: colors[i] for i in range(num_clusters)})
df_pca_2d['color'] = df_pca_2d['color'].fillna('#000000')

plt.figure(figsize=(8, 6))
plt.scatter(df_pca_2d['PC1'], df_pca_2d['PC2'], c=df_pca_2d['color'], alpha=0.6, s=10, label='Casas')
plt.scatter(centroids_pca_2d[:, 0], centroids_pca_2d[:, 1], marker='*', s=250, c='black', label='Centróides')
plt.title("Clusters de Imóveis (PCA 2D) com Centróides")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(True)
plt.show()

# PCA 3D
pca_3d = PCA(n_components=3)
pca_array_3d = pca_3d.fit_transform(X_scaled)
centroids_pca_3d = pca_3d.transform(centroids)

df_pca_3d = pd.DataFrame(data=pca_array_3d, columns=['PC1', 'PC2', 'PC3'])
df_pca_3d['group'] = group
df_pca_3d['color'] = df_pca_3d['group'].map({i: colors[i] for i in range(num_clusters)})
df_pca_3d['color'] = df_pca_3d['color'].fillna('#000000')

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_pca_3d['PC1'], df_pca_3d['PC2'], df_pca_3d['PC3'], c=df_pca_3d['color'], s=20, alpha=0.7, label='Casas')
ax.scatter(centroids_pca_3d[:, 0], centroids_pca_3d[:, 1], centroids_pca_3d[:, 2], c='black', s=300, marker='*', label='Centróides')
ax.set_title("Clusters de Imóveis (PCA 3D) com Centróides")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.legend()
plt.show()

# Agrupamento final no DataFrame original
df['cluster'] = group

# Preço médio por cluster
mean_price_per_cluster = df.groupby('cluster')['Price'].mean().reset_index()
mean_price_per_cluster.columns = ['Cluster', 'Preço Médio']
mean_price_per_cluster = mean_price_per_cluster.sort_values(by='Preço Médio', ascending=False)
print(mean_price_per_cluster)

# Visualização do preço médio por cluster
plt.figure(figsize=(10, 6))
plt.bar(mean_price_per_cluster['Cluster'], mean_price_per_cluster['Preço Médio'], color='skyblue')
plt.title('Preço Médio por Cluster')
plt.xlabel('Cluster')
plt.ylabel('Preço Médio')
plt.xticks(mean_price_per_cluster['Cluster'])
plt.grid(axis='y')
plt.show()

# Média das variáveis numéricas por cluster
mean_attributes_per_cluster = df.groupby('cluster').mean(numeric_only=True).reset_index()
cols_to_show = ['cluster', 'Bedrooms', 'Bathrooms', 'Area', 'Price']
print(mean_attributes_per_cluster[cols_to_show].sort_values(by='Price', ascending=False))