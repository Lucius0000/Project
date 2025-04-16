# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 15:26:19 2025

@author: Lucius
"""

import pandas as pd

# 读取 Stata .dta 文件
df = pd.read_stata(r"C:\Users\Lucius\Desktop\data_changed\summerOly_athletes.dta")

# 筛选出 medal == "Gold" 的行
df_gold = df[df['medal'] == 'Gold']

# 按照 year 和 country 分组，计算 sport_event 的唯一值数量
df_unique_count = df_gold.groupby(['year', 'country'])['sport_event'].nunique().reset_index()

# 重命名列名
df_unique_count.rename(columns={'sport_event': 'true_gold'}, inplace=True)

# 输出结果
print(df_unique_count)

# 输出结果到 Stata .dta 文件
df_unique_count.to_stata('output.dta', write_index=False)
