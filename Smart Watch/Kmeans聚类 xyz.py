# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 23:16:23 2025

@author: Lucius
"""

import os
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# 路径设定
data_dir = r'C:\Users\Lucius\Downloads\2025泰迪杯 赛题\B\B题-全部数据\result_lightgbm_gpu'
output_dir = r'C:\Users\Lucius\Downloads\2025泰迪杯 赛题\B\B题-全部数据\Kmeans_result_xyz'
os.makedirs(output_dir, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

summary_records = []
k_final = 3  # 手动设定K值

weighted_acc_sum = 0.0
total_weight = 0

def find_continuous_sleep(df, min_points=1000, allowed_ratio=0.1):
    """允许波动的连续MET<1检测"""
    condition = df['MET_predicted'] < 1
    group_id = (condition != condition.shift()).cumsum()
    valid_groups = df.groupby(group_id).filter(
        lambda g: g['MET_predicted'].lt(1).mean() >= (1 - allowed_ratio) and len(g) >= min_points
    )
    return valid_groups

for filename in tqdm(os.listdir(data_dir), desc='正在处理文件'):
    if filename.startswith('P') and filename.endswith('.csv'):
        pid = filename.replace('P', '').replace('.csv', '')
        file_path = os.path.join(data_dir, filename)

        df = pd.read_csv(file_path, usecols=['x', 'y', 'z', 'MET_predicted', 'time'], dtype={
            'x': 'float32', 'y': 'float32', 'z': 'float32', 'MET_predicted': 'float32'
        })
        df['acc_mag'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
        df['time'] = pd.to_datetime(df['time'])

        df_sleep = find_continuous_sleep(df)

        if not df_sleep.empty:
            # 聚类基于 x, y, z
            kmeans = KMeans(n_clusters=k_final, random_state=42, n_init=10)
            df_sleep['cluster'] = kmeans.fit_predict(df_sleep[['x', 'y', 'z']]) + 1

            num_sleep_samples = len(df_sleep)
            summary = {
                '志愿者 ID': pid,
                '睡眠总时长（小时）': round(num_sleep_samples / 100 / 3600, 4)
            }

            for i in range(1, k_final + 1):
                cluster_data = df_sleep[df_sleep['cluster'] == i]
                hours = round(len(cluster_data) / 100 / 3600, 4)
                summary[f'睡眠模式{i}总时长（小时）'] = hours
                summary[f'睡眠模式{i}平均x'] = round(cluster_data['x'].mean(), 5)
                summary[f'睡眠模式{i}平均y'] = round(cluster_data['y'].mean(), 5)
                summary[f'睡眠模式{i}平均z'] = round(cluster_data['z'].mean(), 5)
                summary[f'睡眠模式{i}标准差x'] = round(cluster_data['x'].std(), 5)
                summary[f'睡眠模式{i}标准差y'] = round(cluster_data['y'].std(), 5)
                summary[f'睡眠模式{i}标准差z'] = round(cluster_data['z'].std(), 5)

            summary_records.append(summary)

            # 饼图
            counts = df_sleep['cluster'].value_counts().sort_index()
            plt.figure(figsize=(5, 5), dpi=120)
            plt.pie(counts, labels=[f'阶段{i}' for i in counts.index],
                    autopct='%1.1f%%', startangle=140,
                    colors=['#4C72B0', '#55A868', '#C44E52'])
            plt.title(f'志愿者 {pid} 睡眠模式比例', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'sleep_plot_{pid}_pie.png'))
            plt.close()

            # 时间序列图，色标按cluster
            plt.figure(figsize=(10, 4), dpi=120)
            plt.scatter(df_sleep['time'], df_sleep['acc_mag'],
                        c=df_sleep['cluster'], cmap='viridis', s=2, alpha=0.6, edgecolors='none')
            plt.ylim(df_sleep['acc_mag'].min() - 0.01, df_sleep['acc_mag'].max() + 0.01)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            plt.gcf().autofmt_xdate()
            plt.xlabel('时间', fontsize=10)
            plt.ylabel('加速度模值 acc_mag', fontsize=10)
            plt.title(f'志愿者 {pid} 睡眠阶段时间序列', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'sleep_plot_{pid}_timeline.png'))
            plt.close()

            '''
            # PCA二维降维（基于x,y,z）
            pca = PCA(n_components=2)
            reduced = pca.fit_transform(df_sleep[['x', 'y', 'z']])
            plt.figure(figsize=(6, 5), dpi=120)
            plt.scatter(reduced[:, 0], reduced[:, 1], c=df_sleep['cluster'],
                        cmap='viridis', s=8, alpha=0.8)
            plt.xlabel('PCA 主成分 1', fontsize=10)
            plt.ylabel('PCA 主成分 2', fontsize=10)
            plt.title(f'志愿者 {pid} 睡眠模式聚类分布', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'sleep_plot_{pid}_pca.png'))
            plt.close()
            '''

            # 箱线图，x/y/z分别绘制
            for axis in ['x', 'y', 'z']:
                plt.figure(figsize=(6, 5), dpi=120)
                sns.boxplot(x='cluster', y=axis, data=df_sleep, palette='viridis')
                plt.xlabel('睡眠模式')
                plt.ylabel(f'{axis} 加速度值')
                plt.title(f'志愿者 {pid} {axis}值分布（按聚类）', fontsize=12)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'sleep_plot_{pid}_box_{axis}.png'))
                plt.close()

            # 三维可视化
            fig = plt.figure(figsize=(8, 6), dpi=120)
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(df_sleep['x'], df_sleep['y'], df_sleep['z'], c=df_sleep['cluster'], cmap='viridis', s=8, alpha=0.8)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_title(f'志愿者 {pid} 睡眠模式三维聚类分布')
            fig.colorbar(scatter, label='聚类标签')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'sleep_plot_{pid}_3d.png'))
            plt.close()
            
            # SVM 分类验证
            try:
                X = df_sleep[['x', 'y', 'z']].values.astype('float32')
                y = df_sleep['cluster'].values.astype('int32')
            
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42, stratify=y
                )
                svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
                svm.fit(X_train, y_train)
                y_pred = svm.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                summary['SVM准确率'] = round(acc, 4)
                weight = num_sleep_samples
                weighted_acc_sum += acc * weight
                total_weight += weight
                
            except Exception as e:
                summary['SVM准确率'] = '错误'

        del df, df_sleep
        gc.collect()

# 保存统计表
summary_df = pd.DataFrame(summary_records)
summary_df.to_excel(os.path.join(output_dir, 'sleep_summary_xyz.xlsx'), index=False)

# 排除SVM错误项并绘图
valid_summaries = summary_df[summary_df['SVM准确率'] != '错误'].copy()
valid_summaries['SVM准确率'] = valid_summaries['SVM准确率'].astype(float)
valid_summaries = valid_summaries.sort_values(by='SVM准确率', ascending=False)

plt.figure(figsize=(12, 6), dpi=120)
sns.barplot(x='志愿者 ID', y='SVM准确率', data=valid_summaries, palette='viridis')
plt.xticks(rotation=90)
plt.title('各志愿者 SVM 分类准确率排序', fontsize=14)
plt.xlabel('志愿者 ID')
plt.ylabel('准确率')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'svm_accuracy_ranking.png'))
plt.close()

if total_weight > 0:
    overall_weighted_acc = weighted_acc_sum / total_weight
    result_text = f"加权整体 SVM 分类准确率: {overall_weighted_acc:.4f}"
    with open(os.path.join(output_dir, 'svm_accuracy_result.txt'), 'w', encoding='utf-8') as file:
        file.write(result_text)

    with pd.ExcelWriter(os.path.join(output_dir, 'sleep_summary_xyz.xlsx'), engine='openpyxl', mode='a') as writer:
        pd.DataFrame({'加权SVM准确率': [round(overall_weighted_acc, 4)]}).to_excel(writer, index=False, sheet_name='加权SVM')
