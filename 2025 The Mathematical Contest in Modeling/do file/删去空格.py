# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 16:42:54 2025

@author: Lucius
"""

import pandas as pd

# 读取 CSV 文件
df = pd.read_csv(r'C:\Users\Lucius\Desktop\2025_Problem_C_Data\summerOly_athletes.csv')

# 查看原始数据（可选）
print("原始数据：")
print(df.head())

# 删除 'noc' 列的前后空格
df['NOC'] = df['NOC'].str.strip()
df['Team'] = df['Team'].str.strip()

# 保存修改后的数据到新的 CSV 文件
df.to_csv(r'C:\Users\Lucius\Desktop\2025_Problem_C_Data\cleaned_athlete_data.csv', index=False)

# 查看处理后的数据（可选）
print("\n处理后的数据：")
print(df.head())
