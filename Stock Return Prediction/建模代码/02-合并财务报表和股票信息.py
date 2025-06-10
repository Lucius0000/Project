# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 16:43:06 2025

@author: Lucius
"""

import pandas as pd

#df_stock = pd.read_csv('sample（未滚动）.csv', encoding = 'gbk')
df_stock = pd.read_csv('滚动窗口预测2021-2024.csv')
df_tfidf = pd.read_csv('10000_TFIDF_SVD_特征向量.csv')

# 统一股票代码为字符串（保留前导零）
df_stock['A股股票代码_A_StkCd'] = df_stock['A股股票代码_A_StkCd'].astype(str).str.zfill(6)
df_tfidf['A股股票代码_A_StkCd'] = df_tfidf['A股股票代码_A_StkCd'].astype(str).str.zfill(6)

# 解析日期列
df_stock['StockDate'] = pd.to_datetime(df_stock['日期_Date'], errors='coerce')
df_tfidf['TfDate'] = pd.to_datetime(df_tfidf['日期_Date'], errors='coerce', infer_datetime_format=True)

# 清洗日期为空的行
df_stock = df_stock.dropna(subset=['StockDate'])
df_tfidf = df_tfidf.dropna(subset=['TfDate'])

df_all = pd.merge(df_stock, df_tfidf, on='A股股票代码_A_StkCd', suffixes=('_stk', '_tfidf'))

# 仅保留 TfDate <= StockDate 的记录
df_all = df_all[df_all['TfDate'] <= df_all['StockDate']]

# 计算滞后时间
df_all['date_diff'] = (df_all['StockDate'] - df_all['TfDate']).dt.days

# 保留最近一条 TF-IDF 报告
df_all = df_all.sort_values(['A股股票代码_A_StkCd', 'StockDate', 'date_diff'])
df_merged = df_all.groupby(['A股股票代码_A_StkCd', 'StockDate']).first().reset_index()

# 计算滞后月份
df_merged['report_lag_months'] = (
    df_merged['StockDate'].dt.year * 12 + df_merged['StockDate'].dt.month
  - df_merged['TfDate'].dt.year * 12 - df_merged['TfDate'].dt.month
)

# TF-IDF 特征列提取
tfidf_cols = [col for col in df_merged.columns if col.startswith('SVD特征')]
df_merged[tfidf_cols] = df_merged[tfidf_cols].ffill()

# 拆分为上交所/深交所
df_merged['代码'] = df_merged['A股股票代码_A_StkCd']
df_sh = df_merged[df_merged['代码'].str.startswith(('60','601','603','605','688'))].reset_index(drop=True)
df_sz = df_merged[df_merged['代码'].str.startswith(('000','002','300'))].reset_index(drop=True)

print(df_sh[['代码','StockDate','TfDate','report_lag_months']].head())
print(df_sz[['代码','StockDate','TfDate','report_lag_months']].head())
df_sh.to_csv('aa.csv', encoding = 'utf-8-sig', index = False)
df_sz.to_csv('bb.csv', encoding = 'utf-8-sig', index = False)