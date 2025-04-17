# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 19:53:04 2025

@author: Lucius
"""

import pandas as pd

df = pd.read_stata(r"C:\Users\Lucius\Desktop\data_changed\program_changed_sport.dta" )

df1 = df.groupby('sport')['event_count'].sum().reset_index()


total_event_count = df1['event_count'].sum()


df1['weight'] = df1['event_count'] / total_event_count

df1.to_excel("各sport的总event数.xlsx",index = False)