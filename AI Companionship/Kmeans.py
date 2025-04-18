# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 21:41:05 2025

@author: Lucius
"""


import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder

df = pd.read_excel(r"C:\Users\Lucius\Desktop\问卷数据 second.xlsx")  

ohe = OneHotEncoder(drop='first', sparse_output=False) 
label_enc = LabelEncoder()

categorical_features = ['Gender','Identity']
df_encoded = pd.DataFrame(ohe.fit_transform(df[categorical_features]))
df_encoded.columns = ohe.get_feature_names_out(categorical_features)

ordinal_features = ["Age","Spending","Loneliness","Depre"]
scaler_z = StandardScaler()
df[ordinal_features] = scaler_z.fit_transform(df[ordinal_features])

'''
# 期望价格（价格预期）处理：缺失值填 0 + Min-Max 归一化
df['期望价格'] = df['期望价格'].fillna(0)  # 缺失值设为 0
scaler_minmax = MinMaxScaler()
df[['期望价格']] = scaler_minmax.fit_transform(df[['期望价格']])
'''

df2 = df[ordinal_features]
df2 = df2.copy()  
df_final = pd.concat([df2, df_encoded], axis=1)
#df_final.info()
#df_final.to_excel("数据处理.xlsx",index = False)
#df_final.isnull().sum()



import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans



plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False 

K_range = range(1, 11)  
sse = [] 

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_final
    sse.append(kmeans.inertia_)  

plt.figure(figsize=(8, 5))
plt.plot(K_range, sse, marker='o', linestyle='-')
plt.xlabel('簇的个数 K')
plt.ylabel('误差平方和 (SSE)')
plt.title('肘部法确定最佳 K 值')
plt.xticks(K_range)
plt.grid()
plt.show()



from sklearn.metrics import silhouette_score

silhouette_scores = []
K_range = range(2, 11) 

for K in K_range:
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels = kmeans.fit_predict(df_final)
    score = silhouette_score(df_final, labels)
    silhouette_scores.append(score)

plt.figure(figsize=(8, 5))
plt.plot(K_range, silhouette_scores, marker='o', linestyle='-')
plt.xlabel('聚类数 K')
plt.ylabel('轮廓系数')
plt.title('不同 K 值的轮廓系数')
plt.grid()
plt.show()



kmeans = KMeans(n_clusters=3, random_state=42)

kmeans.fit(df_final)

df_final['Cluster'] = kmeans.labels_

print(df_final.head())
df_final.to_excel("聚类结果.xlsx", index=False)


centroids = kmeans.cluster_centers_

unique, counts = np.unique(kmeans.labels_, return_counts=True)
cluster_sizes = dict(zip(unique, counts))

df_centroids = pd.DataFrame(centroids, columns=[f'Feature_{i+1}' for i in range(centroids.shape[1])])
df_centroids.to_excel('聚类中心.xlsx', index=False, sheet_name='Centroids')

df_cluster_sizes = pd.DataFrame(list(cluster_sizes.items()), columns=['Cluster', 'Sample_Count'])
df_cluster_sizes.to_excel('每个簇的样本数量.xlsx', index=False, sheet_name='Cluster Sizes')

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_final.drop('Cluster', axis=1))

plt.figure(figsize=(8, 6))
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=df_final['Cluster'], cmap='viridis', alpha=0.7)
plt.title('KMeans 聚类结果 (PCA_2D降维)')
plt.xlabel('主成分 1')
plt.ylabel('主成分 2')
plt.colorbar(label='Cluster')
plt.show()


pca_3d = PCA(n_components=3)
df_pca_3d = pca_3d.fit_transform(df_final.drop('Cluster', axis=1))

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(df_pca_3d[:, 0], df_pca_3d[:, 1], df_pca_3d[:, 2], c=df_final['Cluster'], cmap='viridis', alpha=0.7)
plt.title('KMeans 聚类结果 (PCA_3D降维)')
ax.set_xlabel('主成分 1')
ax.set_ylabel('主成分 2')
ax.set_zlabel('主成分 3')
plt.colorbar(sc, label='Cluster')
plt.show()



from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42)
df_tsne = tsne.fit_transform(df_final.drop('Cluster', axis=1))

plt.figure(figsize=(8, 6))
plt.scatter(df_tsne[:, 0], df_tsne[:, 1], c=df_final['Cluster'], cmap='viridis', alpha=0.7)
plt.title('KMeans 聚类结果 (t-SNE降维)')
plt.xlabel('t-SNE 维度 1')
plt.ylabel('t-SNE 维度 2')
plt.colorbar(label='Cluster')
plt.show()



import seaborn as sns

centroids = kmeans.cluster_centers_

centroids_df = pd.DataFrame(centroids, columns=df_final.drop('Cluster', axis=1).columns)
plt.figure(figsize=(10, 6))
sns.heatmap(centroids_df, annot=True, cmap='coolwarm', cbar=True)
plt.title('KMeans 聚类中心热力图')
plt.show()



import seaborn as sns
sns.pairplot(df_final, hue='Cluster', palette='viridis', plot_kws={'alpha': 0.7})
plt.suptitle('KMeans 聚类结果 (成对特征散点图)', y=1.02)
plt.show()



