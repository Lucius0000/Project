# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 00:00:12 2025

@author: Lucius
"""

import pandas as pd
import os


# 设置文件夹路径
folder_path_A = r"C:\Users\Lucius\Desktop\数据(1)\分项目获金数据合集"
folder_path_B = r"C:\Users\Lucius\Desktop\数据(1)\2028预测数据获金"  # 替换为你的文件夹路径
output_folder = r"C:\Users\Lucius\Desktop\数据(1)\获金数据独热编码" 
os.makedirs(output_folder, exist_ok=True)

# 获取文件夹中所有.xlsx文件
xlsx_files_A = [f for f in os.listdir(folder_path_A) if f.endswith('.xlsx')]
#print (xlsx_files_A)

xlsx_files_B = [f for f in os.listdir(folder_path_B) if f.endswith('.xlsx')]

pred_file_path = r"C:\Users\Lucius\Desktop\result\pred_2028_sport_forests.xlsx"

all_predictions = pd.DataFrame()

# 定义阈值：非全为0的列数大于该值时使用随机森林
non_zero_column_threshold = 0

# 打开一个文件来保存文本输出
with open(os.path.join(folder_path_B, 'model_output.txt'), 'w') as output_file:

    # 遍历每个文件并处理
    for file in xlsx_files_A:
        
        if file in xlsx_files_B:  # 匹配同名文件并构建完整路径
            file_path_A = os.path.join(folder_path_A, file)
            file_path_B = os.path.join(folder_path_B, file)
            
            # 加载数据
            data = pd.read_excel(file_path_A)  # 替换为实际数据路径
            
            # 示例特征：GDP、人口、历史奖牌数、参赛项目数等
            features = ['noc','sport_gold','year','participant_count','Country','event_count','host','year_count','gold_previous_1','participants_previous_1','gold_previous_2','participants_previous_2','gold_previous_3','participants_previous_3','gold_previous_4','participants_previous_4','gold_previous_5','participants_previous_5','gold_previous_6','participants_previous_6','gold_previous_7','participants_previous_7','gold_previous_8','participants_previous_8','gold_previous_9','participants_previous_9','gold_previous_10','participants_previous_10','gold_previous_11','participants_previous_11','gold_previous_12','participants_previous_12','gold_previous_13','participants_previous_13','gold_previous_14','participants_previous_14','gold_previous_15','participants_previous_15','gold_previous_16','participants_previous_16','gold_previous_17','participants_previous_17','gold_previous_18','participants_previous_18','gold_previous_19','participants_previous_19','gold_previous_20','participants_previous_20','gold_previous_21','participants_previous_21','gold_previous_22','participants_previous_22','gold_previous_23','participants_previous_23','gold_previous_24','participants_previous_24','gold_previous_25','participants_previous_25','gold_previous_26','participants_previous_26','gold_previous_27','participants_previous_27','gold_previous_28','participants_previous_28','gold_previous_29','participants_previous_29']
            target = 'sport_gold'

            # 数据预处理
            X = data[features]
            y = data[target]
            
            X = X.fillna(0)
                        
            # 将 'Country' 列进行独热编码，去掉第一个列（避免虚拟变量陷阱）
            X = pd.get_dummies(X, columns=['Country'], drop_first=False)
            
            # 筛选掉全为 0 的特征列
            X = X.loc[:, (X != 0).any(axis=0)]

            
            print(X.head())
            
            # 更新后的数据可以保存到新的文件夹
            output_file_path = os.path.join(output_folder, file)
            X.to_excel(output_file_path, index=False)
            print("success")
            
            
            
            
            
            