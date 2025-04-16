# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 21:41:05 2025

@author: Lucius
"""


import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder


# 读取数据
df = pd.read_excel(r"C:\Users\Lucius\Desktop\问卷数据 second.xlsx")  

# 处理类别变量
ohe = OneHotEncoder(drop='first', sparse_output=False)  # 兼容新版 sklearn  # drop='first'避免多重共线性
label_enc = LabelEncoder()

# 性别、身份、专业领域 -> 独热编码
categorical_features = ['Gender','Identity']
df_encoded = pd.DataFrame(ohe.fit_transform(df[categorical_features]))
df_encoded.columns = ohe.get_feature_names_out(categorical_features)

# 年龄、月消费额、社交活跃度 -> 标签编码 + Z-score
ordinal_features = ["Age","Spending","Loneliness","Depre"]
scaler_z = StandardScaler()
df[ordinal_features] = scaler_z.fit_transform(df[ordinal_features])

'''
# 期望价格（价格预期）处理：缺失值填 0 + Min-Max 归一化
df['期望价格'] = df['期望价格'].fillna(0)  # 缺失值设为 0
scaler_minmax = MinMaxScaler()
df[['期望价格']] = scaler_minmax.fit_transform(df[['期望价格']])
'''

# 合并所有处理后的特征
df2 = df[ordinal_features]
df2 = df2.copy()  # 先显式复制
df_final = pd.concat([df2, df_encoded], axis=1)
#df_final.info()
#df_final.to_excel("数据处理.xlsx",index = False)
#df_final.isnull().sum()



import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans



plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 选择 K 值的范围
K_range = range(1, 11)  # 1 到 10 个聚类
sse = []  # 存放每个 K 对应的误差平方和

# 计算不同 K 值的 SSE
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_final)  # df_final 是标准化+编码后的数据
    sse.append(kmeans.inertia_)  # inertia_ 就是 SSE

# 画出肘部法曲线
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
K_range = range(2, 11)  # 取 K=2 到 K=10 进行测试

for K in K_range:
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels = kmeans.fit_predict(df_final)
    score = silhouette_score(df_final, labels)
    silhouette_scores.append(score)

# 画出轮廓系数曲线
plt.figure(figsize=(8, 5))
plt.plot(K_range, silhouette_scores, marker='o', linestyle='-')
plt.xlabel('聚类数 K')
plt.ylabel('轮廓系数')
plt.title('不同 K 值的轮廓系数')
plt.grid()
plt.show()




# 初始化KMeans，k=3表示分成3个类
kmeans = KMeans(n_clusters=3, random_state=42)

# 对数据进行聚类
kmeans.fit(df_final)

# 获取每个样本的簇标签
df_final['Cluster'] = kmeans.labels_

# 查看聚类结果
print(df_final.head())

# 你可以将聚类结果保存到新的文件中
df_final.to_excel("聚类结果.xlsx", index=False)


# 获取聚类中心
centroids = kmeans.cluster_centers_

# 获取每个簇的样本数量
unique, counts = np.unique(kmeans.labels_, return_counts=True)
cluster_sizes = dict(zip(unique, counts))

# 将聚类中心保存到Excel
df_centroids = pd.DataFrame(centroids, columns=[f'Feature_{i+1}' for i in range(centroids.shape[1])])
df_centroids.to_excel('聚类中心.xlsx', index=False, sheet_name='Centroids')

# 将每个簇的样本数量保存到Excel
df_cluster_sizes = pd.DataFrame(list(cluster_sizes.items()), columns=['Cluster', 'Sample_Count'])
df_cluster_sizes.to_excel('每个簇的样本数量.xlsx', index=False, sheet_name='Cluster Sizes')


# 可视化（如果数据维度较高，可以选择合适的维度进行可视化）
# 这里我们以前两个主成分为例，进行聚类结果的可视化
from sklearn.decomposition import PCA

# 使用PCA将数据降维至2维
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_final.drop('Cluster', axis=1))

# 将降维后的数据与聚类标签一起绘制
plt.figure(figsize=(8, 6))
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=df_final['Cluster'], cmap='viridis', alpha=0.7)
plt.title('KMeans 聚类结果 (PCA_2D降维)')
plt.xlabel('主成分 1')
plt.ylabel('主成分 2')
plt.colorbar(label='Cluster')
plt.show()



# 使用PCA将数据降维至3维
pca_3d = PCA(n_components=3)
df_pca_3d = pca_3d.fit_transform(df_final.drop('Cluster', axis=1))

# 3D可视化
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 使用scatter绘制3D散点图
sc = ax.scatter(df_pca_3d[:, 0], df_pca_3d[:, 1], df_pca_3d[:, 2], c=df_final['Cluster'], cmap='viridis', alpha=0.7)
plt.title('KMeans 聚类结果 (PCA_3D降维)')
ax.set_xlabel('主成分 1')
ax.set_ylabel('主成分 2')
ax.set_zlabel('主成分 3')
plt.colorbar(sc, label='Cluster')
plt.show()



from sklearn.manifold import TSNE

# 使用t-SNE降维至2维
tsne = TSNE(n_components=2, random_state=42)
df_tsne = tsne.fit_transform(df_final.drop('Cluster', axis=1))

# 可视化 t-SNE 结果
plt.figure(figsize=(8, 6))
plt.scatter(df_tsne[:, 0], df_tsne[:, 1], c=df_final['Cluster'], cmap='viridis', alpha=0.7)
plt.title('KMeans 聚类结果 (t-SNE降维)')
plt.xlabel('t-SNE 维度 1')
plt.ylabel('t-SNE 维度 2')
plt.colorbar(label='Cluster')
plt.show()



import seaborn as sns

# 获取每个簇的聚类中心
centroids = kmeans.cluster_centers_

# 将聚类中心转换为DataFrame
centroids_df = pd.DataFrame(centroids, columns=df_final.drop('Cluster', axis=1).columns)

# 可视化聚类中心的热力图
plt.figure(figsize=(10, 6))
sns.heatmap(centroids_df, annot=True, cmap='coolwarm', cbar=True)
plt.title('KMeans 聚类中心热力图')
plt.show()



import seaborn as sns

# 选择部分重要特征进行成对散点图绘制
sns.pairplot(df_final, hue='Cluster', palette='viridis', plot_kws={'alpha': 0.7})
plt.suptitle('KMeans 聚类结果 (成对特征散点图)', y=1.02)
plt.show()



