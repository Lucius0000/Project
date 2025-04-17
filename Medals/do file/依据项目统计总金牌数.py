# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 22:29:12 2025

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

df_country_total['Pred_Gold'] = np.round(df_country_total['Pred_Gold']).astype(int) 

# 对总奖牌数进行排序（按降序排列）
df_country_total = df_country_total.sort_values(by='Pred_Gold', ascending=False)

# 查看合并后的结果
print("Total Gold by Country with NOC and Year:")
print(df_country_total)

# 保存为 CSV 文件
df_country_total.to_csv('total_gold_by_country_with_noc_year.csv', index=False)