# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 18:46:40 2025

@author: Lucius
"""


import pandas as pd
import numpy as np

df = pd.read_excel(r"C:\Users\Lucius\Desktop\result\pred_2028_gold_sport_forests.xlsx")

df_country_total = df.groupby('Country').agg(
    noc=('noc', 'first'),  # 保留第一个出现的 noc
    Year=('Year', 'first'),  # 保留第一个出现的 Year
    Pred_Gold=('Pred_Gold', 'sum')  # 求和
).reset_index()

# 对总奖牌数进行排序（按降序排列）
df_country_total = df_country_total.sort_values(by='Pred_Gold', ascending=False)

# 查看合并后的结果
print("Total Gold by Country with NOC and Year:")
print(df_country_total)

# 保存为 CSV 文件
df_country_total.to_csv('total_gold_by_country（保留小数）.csv', index=False)