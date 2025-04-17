# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 01:27:04 2025

@author: Lucius
"""

import pandas as pd

# 读取数据
df = pd.read_stata(r"C:\Users\Lucius\Desktop\do file\most_important_event.dta")

# 按 sport 分类，并计算 sport_gold 和 sport_medal 的方差
variance = df.groupby('sport')[['sport_gold', 'sport_medal']].var()

# 输出结果到新的 Stata 文件
variance.to_stata("sport_variance.dta", write_index=True)
