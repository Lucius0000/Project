# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 20:39:18 2025

@author: Lucius
"""

import pandas as pd

df = pd.read_stata(r"C:\Users\Lucius\Desktop\data_changed\medal权重.dta")

df.fillna(0)

columns_to_modify = ['r2','mse','Country','event_count','host','medal_previous_1','medal_previous_10','medal_previous_11','medal_previous_12','medal_previous_13','medal_previous_14','medal_previous_15','medal_previous_16','medal_previous_17','medal_previous_18','medal_previous_19','medal_previous_2','medal_previous_20','medal_previous_21','medal_previous_22','medal_previous_23','medal_previous_24','medal_previous_25','medal_previous_26','medal_previous_27','medal_previous_28','medal_previous_29','medal_previous_3','medal_previous_4','medal_previous_5','medal_previous_6','medal_previous_7','medal_previous_8','medal_previous_9','participant_count','participants_previous_1','participants_previous_10','participants_previous_11','participants_previous_12','participants_previous_13','participants_previous_14','participants_previous_15','participants_previous_16','participants_previous_17','participants_previous_18','participants_previous_19','participants_previous_2','participants_previous_20','participants_previous_21','participants_previous_22','participants_previous_23','participants_previous_24','participants_previous_25','participants_previous_26','participants_previous_27','participants_previous_28','participants_previous_29','participants_previous_3','participants_previous_4','participants_previous_5','participants_previous_6','participants_previous_7','participants_previous_8','participants_previous_9','year_count']

sums = {}

# 使用 for 循环遍历列名并进行操作
for col in columns_to_modify:
    df[col] = df[col] * df['weight']
        
        
for col in columns_to_modify:
    sums[col] = df[col].sum()


sum_df = pd.DataFrame(sums, index=[0])

# 查看修改后的数据
print(df[columns_to_modify].head())

# 保存到新的文件（如果需要）
df.to_stata("modified_performance_medal.dta", write_index=False)
sum_df.to_stata("sum_performance_medal.dta", write_index=False)