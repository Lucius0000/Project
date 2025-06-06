# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 23:08:22 2025

@author: Lucius
"""


import os
import pandas as pd
import numpy as np
import re
import warnings
import gc
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV

warnings.filterwarnings('ignore')

# 路径设置
train_folder = r'C:\Users\Lucius\Downloads\2025泰迪杯 赛题\B\示例数据\附件1'
test_folder = r'C:\Users\Lucius\Downloads\2025泰迪杯 赛题\B\示例数据\附件2'
result_folder = r'C:\Users\Lucius\Downloads\2025泰迪杯 赛题\B\B题-全部数据\result_rf'
os.makedirs(result_folder, exist_ok=True)

# 加载元数据
metadata1 = pd.read_csv(os.path.join(train_folder, 'Metadata1.csv'))
metadata2 = pd.read_csv(os.path.join(test_folder, 'Metadata2.csv'))

# 年龄标签编码
age_mapping = {'18-29': 0, '30-37': 1, '38-52': 2, '53+': 3}
metadata1['age'] = metadata1['age'].map(age_mapping)
metadata2['age'] = metadata2['age'].map(age_mapping)

# 提取 MET 数值
def extract_met_value(met_str):
    match = re.search(r'MET\s*([\d\.]+)', met_str)
    return float(match.group(1)) if match else None

# 构造训练数据
train_data_list = []
files_train = [f for f in os.listdir(train_folder) if f.startswith('P') and f.endswith('.csv')]
for i, file in enumerate(files_train, 1):
    df = pd.read_csv(os.path.join(train_folder, file))
    df = df.head(2000)
    pid = file[:-4]
    if 'annotation' in df.columns:
        df = df.dropna(subset=['annotation'])
        df[['label', 'MET']] = df['annotation'].str.rsplit(';', n=1, expand=True)
        df['MET'] = df['MET'].apply(extract_met_value)
        df = df.dropna(subset=['MET'])

        meta = metadata1[metadata1['pid'] == pid].iloc[0]
        df['age'] = meta['age']
        df['sex'] = 1 if meta['sex'] == 'M' else 0

        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df['interval_15min'] = df['time'].dt.hour * 4 + (df['time'].dt.minute // 15)
        df['dayofweek'] = df['time'].dt.dayofweek
        df['acc'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)

        train_data_list.append(df[['age', 'sex', 'interval_15min', 'dayofweek', 'acc', 'MET']])

    print(f"训练数据读取进度: {i}/{len(files_train)} ({(i/len(files_train)*100):.2f}%)")
    del df
    gc.collect()

train_data = pd.concat(train_data_list, ignore_index=True)
train_data.reset_index(drop=True, inplace=True)
train_data.to_feather(r'D:\train_data.feather')

del train_data_list
gc.collect()

# 读取 feather 格式的数据
train_data = pd.read_feather(r'D:\train_data.feather')
# 构建特征和标签
X_train = train_data[['age', 'sex', 'interval_15min', 'dayofweek', 'acc']].astype('float32')
y_train = train_data['MET'].astype('float32')
del train_data
gc.collect()

# 随机森林参数搜索
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=10, cv=3, verbose=1, random_state=42, n_jobs=-1)
search.fit(X_train, y_train)
rf_best = search.best_estimator_

# 模型评估
y_pred_train = rf_best.predict(X_train)
r2 = r2_score(y_train, y_pred_train)
mse = mean_squared_error(y_train, y_pred_train)
print(f"训练集 R² 分数: {r2:.4f}")
print(f"训练集 MSE: {mse:.4f}")

del X_train, y_train, y_pred_train
gc.collect()

# 保存特征重要性
importances = rf_best.feature_importances_
feature_names = ['age', 'sex', 'interval_15min', 'dayofweek', 'acc']
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
importance_df.to_csv(os.path.join(result_folder, 'feature_importance.csv'), index=False)
print("特征重要性已保存。")

# MET 分类函数
def classify_MET(met):
    if met >= 6.0: return '高强度'
    elif 3.0 <= met < 6.0: return '中等强度'
    elif 1.6 <= met < 3.0: return '低强度'
    elif 1.0 <= met < 1.6: return '静态行为'
    else: return '睡眠'

# 测试数据预测
files_test = [f for f in os.listdir(test_folder) if f.startswith('P') and f.endswith('.csv')]
summary_list = []
for i, file in enumerate(files_test, 1):
    df_test = pd.read_csv(os.path.join(test_folder, file))
    pid = file[:-4]
    meta = metadata2[metadata2['pid'] == pid].iloc[0]
    df_test['age'] = meta['age']
    df_test['sex'] = 1 if meta['sex'] == 'M' else 0

    df_test['time'] = pd.to_datetime(df_test['time'], errors='coerce')
    df_test['interval_15min'] = df_test['time'].dt.hour * 4 + (df_test['time'].dt.minute // 15)
    df_test['dayofweek'] = df_test['time'].dt.dayofweek
    df_test['acc'] = np.sqrt(df_test['x']**2 + df_test['y']**2 + df_test['z']**2)

    features = df_test[['age', 'sex', 'interval_15min', 'dayofweek', 'acc']].astype('float32')
    df_test['MET_predicted'] = rf_best.predict(features)
    df_test['MET_category'] = df_test['MET_predicted'].apply(classify_MET)

    df_test.to_csv(os.path.join(result_folder, file), index=False)

    counts = df_test['MET_category'].value_counts().to_dict()
    duration_hours = {k: round(v / 100 / 3600, 4) for k, v in counts.items()}
    duration_hours['file'] = file
    summary_list.append(duration_hours)

    print(f"测试数据预测进度: {i}/{len(files_test)} ({(i/len(files_test)*100):.2f}%) - 完成 {file}")

    del df_test, features
    gc.collect()

# 汇总统计
summary_df = pd.DataFrame(summary_list).fillna(0)
summary_df.to_csv(os.path.join(result_folder, 'MET_category_summary.csv'), index=False)
print("\n✅ 所有预测完成，统计文件已保存。")
