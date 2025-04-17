# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 23:34:47 2025

@author: Lucius
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
import matplotlib.pyplot as plt


# 设置文件夹路径
folder_path = r"C:\Users\Lucius\Desktop\随机森林数据(1)\分项目获奖数据合集"  # 替换为你的文件夹路径

# 获取文件夹中所有.xlsx文件
xlsx_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
print (xlsx_files)

# 打开一个文件来保存文本输出
with open(os.path.join(folder_path, 'model_output.txt'), 'w') as output_file:

    # 遍历每个文件并处理
    for file in xlsx_files:
        # 构建文件的完整路径
        file_path = os.path.join(folder_path, file)
        
        # 加载数据
        data = pd.read_excel(file_path)  # 替换为实际数据路径

        # 示例特征：GDP、人口、历史奖牌数、参赛项目数等
        features = ['participant_count','host','participation_count','gold_previous_1','participants_previous_1','gold_previous_2','participants_previous_2','gold_previous_3','participants_previous_3','gold_previous_4','participants_previous_4','gold_previous_5','participants_previous_5','gold_previous_6','participants_previous_6','gold_previous_7','participants_previous_7','gold_previous_8','participants_previous_8','gold_previous_9','participants_previous_9','gold_previous_10','participants_previous_10','gold_previous_11','participants_previous_11','gold_previous_12','participants_previous_12','gold_previous_13','participants_previous_13','gold_previous_14','participants_previous_14','gold_previous_15','participants_previous_15','gold_previous_16','participants_previous_16','gold_previous_17','participants_previous_17','gold_previous_18','participants_previous_18','gold_previous_19','participants_previous_19','gold_previous_20','participants_previous_20','gold_previous_21','participants_previous_21','gold_previous_22','participants_previous_22','gold_previous_23','participants_previous_23','gold_previous_24','participants_previous_24','gold_previous_25','participants_previous_25','gold_previous_26','participants_previous_26','gold_previous_27','participants_previous_27','gold_previous_28','participants_previous_28','gold_previous_29','participants_previous_29','gold_previous_30','participants_previous_30']
        target = 'sport_medal'

        # 数据预处理
        X = data[features]
        y = data[target]

        # 将分类特征编码（如果有需要）
        #X = pd.get_dummies(X, columns=['Host_Country'], drop_first=True)

        # 数据集划分
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 初始化随机森林模型
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

        # 训练模型
        rf_model.fit(X_train, y_train)

        # 模型预测
        y_pred = rf_model.predict(X_test)

        # 计算评价指标
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"均方误差 (MSE): {mse:.2f}")
        print(f"R^2 值: {r2:.2f}")
        
        # 输出文本到文件
        output_file.write(f"文件: {file}\n")
        output_file.write(f"均方误差 (MSE): {mse:.2f}\n")
        output_file.write(f"R^2 值: {r2:.2f}\n\n")
        
        
        import matplotlib.pyplot as plt
        from matplotlib import rcParams

        # 设置支持中文的字体
        rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
        rcParams['axes.unicode_minus'] = False  # 防止负号显示问题

        # 提取特征重要性
        feature_importances = rf_model.feature_importances_
        sorted_indices = feature_importances.argsort()

        # 绘制特征重要性条形图
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(sorted_indices)), feature_importances[sorted_indices], align='center')
        plt.yticks(range(len(sorted_indices)), [X.columns[i] for i in sorted_indices])
        plt.xlabel("特征重要性")
        plt.title("随机森林特征重要性（优化前）")
        #plt.show()

    
        
        # 保存图像
        img_output_path = os.path.join(folder_path, f"{file}_feature_importance_formal.png")
        plt.savefig(img_output_path)

        # 清除当前的图形，准备下一个图形
        plt.clf()

        print(f"Feature importance plot saved as {img_output_path}")