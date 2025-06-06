# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 22:07:43 2025

@author: Lucius
"""

import os
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm

data_dir = r'C:\Users\Lucius\Downloads\2025泰迪杯 赛题\B\B题-全部数据\result_lightgbm_gpu'
output_dir = r'C:\Users\Lucius\Downloads\2025泰迪杯 赛题\B\B题-全部数据\Kmeans_result'
os.makedirs(output_dir, exist_ok=True)

all_sleep_features = []

for filename in tqdm(os.listdir(data_dir), desc='读取并提取特征'):
    if filename.startswith('P') and filename.endswith('.csv'):
        file_path = os.path.join(data_dir, filename)
        df = pd.read_csv(file_path, usecols=['x', 'y', 'z', 'MET_predicted'], dtype={
            'x': 'float32', 'y': 'float32', 'z': 'float32', 'MET_predicted': 'float32'
        })
        df['acc_mag'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
        df_sleep = df[df['MET_predicted'] < 1]
        all_sleep_features.append(df_sleep[['x', 'y', 'z', 'acc_mag']])
        del df, df_sleep
        gc.collect()

all_sleep_data = pd.concat(all_sleep_features, ignore_index=True)

inertia = []
k_range = range(2, 6)

for k in tqdm(k_range, desc='计算KMeans误差'):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(all_sleep_data)
    inertia.append(kmeans.inertia_)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(7, 5), dpi=120)
plt.plot(k_range, inertia, marker='o', color='#2c7bb6', linewidth=2)
plt.xticks(k_range)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel('K值', fontsize=12)
plt.ylabel('簇内误差平方和', fontsize=12)
plt.title('Elbow法确定最佳K值', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'elbow_plot.png'))
plt.show()
plt.close()

del all_sleep_features, all_sleep_data
gc.collect()
