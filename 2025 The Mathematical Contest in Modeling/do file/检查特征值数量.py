# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 23:41:41 2025

@author: Lucius
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.model_selection import GridSearchCV


# 设置文件夹路径
folder_path_A = r"C:\Users\Lucius\Desktop\数据(1)\分项目获金数据合集"
folder_path_B = r"C:\Users\Lucius\Desktop\数据(1)\2028预测数据获金"  # 替换为你的文件夹路径

# 获取文件夹中所有.xlsx文件
xlsx_files_A = [f for f in os.listdir(folder_path_A) if f.endswith('.xlsx')]
#print (xlsx_files_A)

xlsx_files_B = [f for f in os.listdir(folder_path_B) if f.endswith('.xlsx')]

pred_file_path = r"C:\Users\Lucius\Desktop\result\pred_2028_sport_forests.xlsx"

all_predictions = pd.DataFrame()

# 定义阈值：非全为0的列数大于该值时使用随机森林
non_zero_column_threshold = 6

# 打开一个文件来保存文本输出
with open(os.path.join(folder_path_A, 'model_output.txt'), 'w') as output_file:

    # 遍历每个文件并处理
    for file in xlsx_files_A:
        
        if file in xlsx_files_B:  # 匹配同名文件并构建完整路径
            file_path_A = os.path.join(folder_path_A, file)
            file_path_B = os.path.join(folder_path_B, file)
            
            # 加载数据
            data = pd.read_excel(file_path_A)  # 替换为实际数据路径
            
            if data.shape[0] < 10:
                #print(f"文件 {file} 的样本数小于10，跳过使用随机森林模型")
                #output_file.write(f"文件 {file} 的样本数小于10，跳过使用随机森林模型\n")
                continue  # 跳过当前文件，进入下一个文件

            #以下是随机森林代码

            # 示例特征：GDP、人口、历史奖牌数、参赛项目数等
            features = ['participant_count','event_count','host','year_count','gold_previous_1','participants_previous_1','gold_previous_2','participants_previous_2','gold_previous_3','participants_previous_3','gold_previous_4','participants_previous_4','gold_previous_5','participants_previous_5','gold_previous_6','participants_previous_6','gold_previous_7','participants_previous_7','gold_previous_8','participants_previous_8','gold_previous_9','participants_previous_9','gold_previous_10','participants_previous_10','gold_previous_11','participants_previous_11','gold_previous_12','participants_previous_12','gold_previous_13','participants_previous_13','gold_previous_14','participants_previous_14','gold_previous_15','participants_previous_15','gold_previous_16','participants_previous_16','gold_previous_17','participants_previous_17','gold_previous_18','participants_previous_18','gold_previous_19','participants_previous_19','gold_previous_20','participants_previous_20','gold_previous_21','participants_previous_21','gold_previous_22','participants_previous_22','gold_previous_23','participants_previous_23','gold_previous_24','participants_previous_24','gold_previous_25','participants_previous_25','gold_previous_26','participants_previous_26','gold_previous_27','participants_previous_27','gold_previous_28','participants_previous_28','gold_previous_29','participants_previous_29']
            target = 'sport_gold'

            # 数据预处理
            X = data[features]
            y = data[target]
            
            X = X.fillna(0)
            
            # 筛选掉全为 0 的特征列
            X = X.loc[:, (X != 0).any(axis=0)]
            
            new_features = list(X.columns)
            #print(set(new_features))
            
            # 计算非全为 0 的列数
            non_zero_column_count = X.shape[1]
            output_file.write(f"文件: {file}, 非全为 0 的列数: {non_zero_column_count}")
            #print(f"文件: {file}, 非全为 0 的列数: {non_zero_column_count}")

            
            # 检查被移除的特征列
            '''
            removed_features = set(features) - set(X.columns)
            if removed_features:
                print(f"被移除的特征: {removed_features}")
                output_file.write(f"文件: {file}\n")
                output_file.write(f"被移除的特征: {removed_features}\n")
                '''
    
            # 根据非全为 0 的列数选择模型
            if non_zero_column_count > non_zero_column_threshold:
                pass
                #output_file.write(f"非全为 0 的列数超过阈值 {non_zero_column_threshold}，使用随机森林")
                
            else:
                print(f"文件: {file},非全为 0 的列数未超过阈值 {non_zero_column_threshold}，转用其他模型")
                output_file.write(f"文件: {file},非全为 0 的列数未超过阈值 {non_zero_column_threshold}，转用其他模型")