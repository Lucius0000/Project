# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 22:12:40 2025

@author: Lucius
"""

import pandas as pd
df = pd.read_excel(r'C:\Users\Lucius\Desktop\变量.xlsx')
#print (df.head())

from sklearn.preprocessing import OneHotEncoder
noc = df["NOC"].values.reshape(-1,1)
encoder = OneHotEncoder(sparse_output=False)
NOC_encode = encoder.fit_transform(noc)
print (NOC_encode)