# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 20:29:41 2025

@author: Lucius
"""

import pandas as pd
df = pd.read_excel(r'C:\Users\Lucius\Desktop\变量.xlsx')
print (df.head())
c_encode = pd.get_dummies(df["NOC"])
print (c_encode.head())