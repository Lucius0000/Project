# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 20:10:46 2025

@author: Lucius
"""

import pandas as pd

df = pd.read_excel(r"C:\Users\Lucius\Desktop\performance_2028_medal_sport_forests_dealt.xlsx")

df_pivoted = df.pivot_table(index=['sport', 'r2', 'mse'], columns='feature', values='feature_importance', aggfunc='first')

# 将 pivot 后的索引转换为列（包括 'sport', 'r2', 'mse' 列），即保留这些列
df_pivoted = df_pivoted.reset_index()

df_pivoted.to_excel("features_performance_2028_medal.xlsx", index=False)