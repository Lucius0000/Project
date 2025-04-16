# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 16:04:31 2025

@author: Lucius
"""

import pandas as pd

# 假设你已经加载了 DataFrame，名为 df
# 例如：
df = pd.read_stata(r"C:\Users\Lucius\Desktop\data_changed\participant_medal.dta"  )

# 先筛选出 sport 为 'Baseball' 和 'Softball' 的数据
df_filtered = df[df['sport'].isin(['Artistic Gymnastics','Gymnastics','Rhythmic Gymnastics','Trampoline Gymnastics'])]

# 按 noc 和 year 聚合，求 sport_medal 的总和
df_aggregated = df_filtered.groupby(['noc', 'year','Country'], as_index=False).agg({
    "sport_gold":'sum',
    "sport_medal":"sum",
    "participant_count":"sum",
    #'event_count':'sum'
    })

# 添加一个新的列 'sport'，将其值设置为 'Baseball and Softball'
df_aggregated['sport'] = 'Gymnastics1'

# 将新数据追加到原始 DataFrame 中
df = pd.concat([df, df_aggregated], ignore_index=True)

#~非运算符
df = df[~df['sport'].isin(['Artistic Gymnastics','Rhythmic Gymnastics','Gymnastics','Trampoline Gymnastics'])]

# 查看结果
print(df)

df.to_stata("300.dta",write_index = False)