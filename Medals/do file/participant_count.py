# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 17:07:19 2025

@author: Lucius
"""

import pandas as pd

df = pd.read_stata(r"C:\Users\Lucius\Desktop\data_changed\summerOly_athletes.dta")

df_name = df.groupby(['year','noc','sport'])['name'].count().reset_index()

df_name.rename(columns={'name': 'participant_count'}, inplace=True)


print(df_name.head())

df_name.to_stata("participant_count.dta",write_index=False)