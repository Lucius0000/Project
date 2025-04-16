# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 19:12:29 2025

@author: Lucius
"""

import pandas as pd

df = pd.read_excel(r"C:\Users\Lucius\Desktop\变量(2).xlsx")

year01 = [2016,2020,2024]

df = df[df['Year'].isin(year01)]

df2028 = df.groupby(['Country']).agg({
    'Gold':'mean',
    'Total':'mean',
    'num_participants':'mean',
    'num_sports':'mean',
    'num_events':'mean'
    })

df2028.to_excel('2028_8.xlsx',index = True)