# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 19:12:29 2025

@author: Lucius
"""

import pandas as pd

df = pd.read_excel(r"C:\Users\Lucius\Desktop\变量.xlsx")

df_medal = df.groupby('Country')[['Gold','Total']].sum().reset_index()

year01 = [2016,2020,2024]

df = df[df['Year'].isin(year01)]

df_event = df.groupby(['Country']).agg({
    'num_participants':'mean',
    'num_sports':'mean',
    'num_events':'mean'
    })

df2028 = pd.merge(df_medal,df_event,on='Country',how='inner')

df2028['year']=2028

df2028.to_excel('2028_all.xlsx',index = True)