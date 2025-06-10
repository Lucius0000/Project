# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 19:13:00 2025

@author: Lucius
"""

import pandas as pd

df1 = pd.read_excel(r"C:\Users\Lucius\Desktop\利润表(清洗).xlsx")
df2 = pd.read_excel(r"C:\Users\Lucius\Desktop\资产负债表(清洗).xlsx")
df3 = pd.read_excel(r'C:/Users/Lucius/Desktop/现金流量表.xlsx')

df1['exists_profit'] = 1
df2['exists_assest'] = 1
df3['exists_cash'] = 1

merge_keys = ['A股股票代码_A_StkCd', '信息发布日期_InfoPubDt']

merged = pd.merge(df1, df2, on=merge_keys, how='outer')
merged = pd.merge(merged, df3, on=merge_keys, how='outer')

merged['exists_profit'] = merged['exists_profit'].fillna(0)
merged['exists_assest'] = merged['exists_assest'].fillna(0)
merged['exists_cash'] = merged['exists_cash'].fillna(0)

total_rows = len(merged)
missing_obs_profit = (merged['exists_profit'] == 0).sum()
missing_obs_assest = (merged['exists_assest'] == 0).sum()
missing_obs_cash = (merged['exists_cash'] == 0).sum()
complete_rows = ((merged['exists_profit'] == 1) & 
                 (merged['exists_assest'] == 1) & 
                 (merged['exists_cash'] == 1)).sum()

print(f"合并后的总观测数：{total_rows}")
print(f"三张表数据都存在的记录数：{complete_rows}")
print("合并后记录中，缺少各来源表数据的行数：")
print(f"缺少利润表数据的记录数：{missing_obs_profit}")
print(f"缺少资产负债表数据的记录数：{missing_obs_assest}")
print(f"缺少现金流量表数据的记录数：{missing_obs_cash}")

merged.to_excel(r"C:\Users\Lucius\Desktop\财务报表2.xlsx")

