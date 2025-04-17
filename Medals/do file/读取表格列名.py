# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 22:58:48 2025

@author: Lucius
"""

import pandas as pd

# 读取Excel文件
df = pd.read_stata(r"C:\Users\Lucius\Desktop\data_changed\medal权重.dta")

# 获取列名
columns = df.columns

# 将列名连接成字符串，使用 + 连接符
column_string = ','.join([f"'{col}'" for col in columns])

# 输出拼接后的字符串
print(column_string)
