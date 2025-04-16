# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 15:55:50 2025

@author: Lucius
"""

import pandas as pd

df = pd.read_stata(r"C:\Users\Lucius\Desktop\data_changed\summerOly_athletes.dta")

df_gold = df[df['medal'] == "Gold"]

df_gold = df_gold.groupby(['year', 'noc', 'sport'])['sport_event'].nunique().reset_index()

df_gold.rename(columns={'sport_event': 'sport_gold'}, inplace=True)




df_silver = df[df['medal'] == "Silver"]

df_silver = df_silver.groupby(['year', 'noc', 'sport'])['sport_event'].nunique().reset_index()

df_silver.rename(columns={'sport_event': 'sport_silver'}, inplace=True)




df_bronze = df[df['medal'] == "Bronze"]

df_bronze = df_bronze.groupby(['year', 'noc', 'sport'])['sport_event'].nunique().reset_index()

df_bronze.rename(columns={'sport_event': 'sport_bronze'}, inplace=True)




df_combined = pd.merge(df_gold, df_silver, on=['year', 'noc', 'sport'], how='outer')
df_combined = pd.merge(df_combined, df_bronze, on=['year', 'noc', 'sport'], how='outer')

# 填充 NaN 值为 0，防止 NaN 导致计算错误
df_combined.fillna(0, inplace=True)

df_combined['sport_total'] = df_combined['sport_gold'] + df_combined['sport_silver'] + df_combined['sport_bronze']

print(df_combined.head())


df_combined.to_stata('sport_medal.dta', write_index=False)