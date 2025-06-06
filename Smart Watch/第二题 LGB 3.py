# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 21:14:37 2025

@author: Lucius
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_squared_error
import re
import warnings
import matplotlib.pyplot as plt
from datetime import datetime
import time
import gc  # 用于主动内存释放

#warnings.filterwarnings('ignore')

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def extract_met_value(met_str):
    match = re.search(r'MET\s*([\d\.]+)', met_str)
    return float(match.group(1)) if match else None

def classify_MET(met):
    if met >= 6.0:
        return '高强度'
    elif 3.0 <= met < 6.0:
        return '中等强度'
    elif 1.6 <= met < 3.0:
        return '低强度'
    elif 1.0 <= met < 1.6:
        return '静态行为'
    else:
        return '睡眠'

# === 路径设置 ===
train_folder = r'C:\Users\Lucius\Downloads\2025泰迪杯 赛题\B\示例数据\附件1'
test_folder = r'C:\Users\Lucius\Downloads\2025泰迪杯 赛题\B\示例数据\附件2'
result_folder = r'C:\Users\Lucius\Downloads\2025泰迪杯 赛题\B\B题-全部数据\result_lightgbm_gpu'
os.makedirs(result_folder, exist_ok=True)

metadata1 = pd.read_csv(os.path.join(train_folder, 'Metadata1.csv'))
metadata2 = pd.read_csv(os.path.join(test_folder, 'Metadata2.csv'))

age_mapping = {
    '18-29': 0, '30-37': 1, '38-52': 2, '53+': 3
}
metadata1['age'] = metadata1['age'].map(age_mapping)
metadata2['age'] = metadata2['age'].map(age_mapping)

# === 训练数据加载 ===
log("开始加载训练数据...")

# 精简数据类型，强制转换为 float32 以节省内存
dtypes = {
    'x': 'float32',
    'y': 'float32',
    'z': 'float32',
    'time': 'str',  # 'time' 列先读取为字符串，再转为 datetime
}

train_data_list = []

for file in os.listdir(train_folder):
    if file.startswith('P') and file.endswith('.csv'):
        df = pd.read_csv(os.path.join(train_folder, file), dtype=dtypes)
        pid = file[:-4]

        if 'annotation' in df.columns:
            df = df.dropna(subset=['annotation'])
            df[['label', 'MET']] = df['annotation'].str.rsplit(';', n=1, expand=True)
            df['MET'] = df['MET'].apply(extract_met_value)
            df = df.dropna(subset=['MET'])
            df = df.drop(columns=['annotation']) 

            meta = metadata1[metadata1['pid'] == pid].iloc[0]
            df['age'] = meta['age']
            df['sex'] = 1 if meta['sex'] == 'M' else 0

            df['time'] = pd.to_datetime(df['time'], errors='coerce')
            df['quarter_hour_index'] = (df['time'].dt.hour * 4 + df['time'].dt.minute // 15) / 20
            df['dayofweek'] = df['time'].dt.dayofweek
            df = df.drop(columns = ["time"])

            # 计算加速度强度 acc_mag = sqrt(x^2 + y^2 + z^2)
            df['acc_mag'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)

            train_data_list.append(df[['x', 'y', 'z', 'acc_mag', 'age', 'sex', 'quarter_hour_index', 'dayofweek', 'MET']])

# 拼接训练数据
train_data = pd.concat(train_data_list, ignore_index=True)
log(f"训练数据加载完成，共 {train_data.shape[0]} 条记录")

# 内存优化2：主动释放未使用变量
del train_data_list, df
gc.collect()

X_train = train_data.drop(columns='MET').astype(np.float32)
y_train = train_data['MET']
del train_data  # 内存优化2：训练集释放
gc.collect()

# 保存成二进制文件
lgb_train_bin = r'D:\train_data.bin'
train_dataset = lgb.Dataset(X_train, label=y_train)
if os.path.exists(lgb_train_bin):
    os.remove(lgb_train_bin)
train_dataset.save_binary(lgb_train_bin)

# 再次释放内存
#del X_train, y_train
#gc.collect()

train_dataset = lgb.Dataset(lgb_train_bin)

# === 检测 GPU ===
log("检测是否可以使用GPU...")
gpu_supported = False
try:
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'learning_rate': 0.05,
        'verbose': -1,
        'max_bin': 255,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1
    }
    _ = lgb.train(params, train_dataset, num_boost_round=1)
    gpu_supported = True
    log("成功启用 GPU 加速")
except:
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'verbose': -1,
        'max_bin': 255,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1
    }
    log("未能使用 GPU，将使用 CPU 训练")

# === 交叉验证 ===
log("开始交叉验证调参...")

from lightgbm import early_stopping, log_evaluation

cv_results = lgb.cv(
    params,
    train_dataset,
    num_boost_round=100,
    nfold=5,
    stratified=False,
    seed=42,
    callbacks=[
        early_stopping(30),
        log_evaluation(100)
    ]
)

# 自动找出 mean 值的 key（防止 rmse-mean/l2-mean 不一致报错）
mean_key = next(k for k in cv_results if k.endswith('-mean'))
best_iter = len(cv_results[mean_key])
log(f"最优迭代次数：{best_iter}，指标为：{mean_key}")

# === 模型训练 ===
log("开始训练最终模型...")
model = lgb.train(params, train_dataset, num_boost_round=best_iter)
log("模型训练完成")

# === 模型评估（R²） ===
log("开始训练集 R² 评估...")

# 使用原始数据进行评估
X_train_eval = X_train  # 直接使用原始 X_train
y_train_eval = y_train  # 直接使用原始 y_train

y_pred_train = model.predict(X_train_eval)
r2 = r2_score(y_train_eval, y_pred_train)
log(f"训练集 R² 得分为：{r2:.5f}")

# === 特征重要性 ===
log("输出特征重要性...")
feature_importance = pd.DataFrame({
    'Feature': model.feature_name(),
    'Importance': model.feature_importance()
}).sort_values(by='Importance', ascending=False)

feature_importance.to_csv(os.path.join(result_folder, 'feature_importance.csv'), index=False)
plt.figure(figsize=(10, 6))
lgb.plot_importance(model, max_num_features=10)
plt.title('Top 10 Feature Importances')
plt.tight_layout()
plt.savefig(os.path.join(result_folder, 'feature_importance.png'))
plt.show()
plt.close()

# === 测试集预测 ===
log("开始预测测试集...")
summary_list = []

for file in os.listdir(test_folder):
    if file.startswith('P') and file.endswith('.csv'):
        df_test = pd.read_csv(os.path.join(test_folder, file), dtype=dtypes)
        pid = file[:-4]

        meta = metadata2[metadata2['pid'] == pid].iloc[0]
        df_test['age'] = meta['age']
        df_test['sex'] = 1 if meta['sex'] == 'M' else 0

        df_test['time'] = pd.to_datetime(df_test['time'], errors='coerce')
        df_test['quarter_hour_index'] = df_test['time'].dt.hour * 4 + df_test['time'].dt.minute // 15
        df_test['dayofweek'] = df_test['time'].dt.dayofweek

        # 计算加速度强度
        df_test['acc_mag'] = np.sqrt(df_test['x']**2 + df_test['y']**2 + df_test['z']**2)

        features = df_test[['x', 'y', 'z', 'acc_mag', 'age', 'sex', 'quarter_hour_index', 'dayofweek']]
        df_test['MET_predicted'] = model.predict(features)

        df_test['MET_category'] = df_test['MET_predicted'].apply(classify_MET)
        df_test.to_csv(os.path.join(result_folder, file), index=False)

        counts = df_test['MET_category'].value_counts().to_dict()
        duration_hours = {key: round(value / 100 / 3600, 4) for key, value in counts.items()}
        duration_hours['file'] = file
        summary_list.append(duration_hours)

        log(f"预测完成：{file}")
        del df_test, features
        gc.collect()

# 输出汇总表
summary_df = pd.DataFrame(summary_list).fillna(0)
summary_df.to_csv(os.path.join(result_folder, 'MET_category_summary.csv'), index=False)
log("所有预测完成，统计结果已保存。")
