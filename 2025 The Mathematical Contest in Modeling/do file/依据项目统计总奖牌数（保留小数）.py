# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 18:47:50 2025

@author: Lucius
"""


import pandas as pd
import numpy as np

df = pd.read_excel(r"C:\Users\Lucius\Desktop\result\pred_2028_medal_sport_forests.xlsx")

df_country_total = df.groupby('Country').agg(
    noc=('noc', 'first'),  # 保留第一个出现的 noc
    Year=('Year', 'first'),  # 保留第一个出现的 Year
    Pred_medal=('Pred_medal', 'sum')  # 求和
).reset_index()


# 对总奖牌数进行排序（按降序排列）
df_country_total = df_country_total.sort_values(by='Pred_medal', ascending=False)

# 查看合并后的结果
print("Total medal by Country with NOC and Year:")
print(df_country_total)

# 保存为 CSV 文件
df_country_total.to_csv('total_medal_by_country（保留小数）.csv', index=False)