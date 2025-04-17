# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 18:52:36 2024

@author: Lucius
"""


import pandas as pd
import numpy as np
import itertools
from joblib import Parallel, delayed



attach_1 = "附件1.xlsx"
attach_2 = "附件2.xlsx"
sheet_1 = "乡村的现有耕地"
sheet_2 = "乡村种植的农作物"
sheet_3 = "2023年的农作物种植情况"
sheet_4 = "2023年统计的相关数据"

df_sheet_1 = pd.read_excel(attach_1, sheet_name=sheet_1)
df_sheet_2 = pd.read_excel(attach_1, sheet_name=sheet_2)
df_sheet_3 = pd.read_excel(attach_2, sheet_name=sheet_3)
df_sheet_4 = pd.read_excel(attach_2, sheet_name=sheet_4)

df_sheet_2=df_sheet_2.drop(index=41).drop(index=42).drop(index=43).drop(index=44)
df_sheet_4_cleaned = df_sheet_4.drop([107, 108, 109], axis=0)


df_sheet_4_cleaned['地块类型'] = df_sheet_4_cleaned['地块类型'].str.strip()
crop_land_mapping = df_sheet_4_cleaned[['作物编号', '地块类型']].dropna().drop_duplicates()
crop_land_mapping_grouped = crop_land_mapping.groupby('作物编号')['地块类型'].apply(lambda x: ','.join(x.unique())).reset_index()
df_sheet_2_updated = df_sheet_2.copy()
df_sheet_2_updated = df_sheet_2_updated.merge(crop_land_mapping_grouped, on='作物编号', how='left')
df_sheet_2_updated['种植耕地'] = df_sheet_2_updated['地块类型']
df_sheet_2_updated = df_sheet_2_updated.drop(columns=['地块类型'])
df_sheet_2_updated['种植耕地'] = df_sheet_2_updated['种植耕地'].str.replace(r'\s+', '', regex=True)
df_sheet_2_updated['说明'] = df_sheet_2_updated['说明'].str.strip()
df_sheet_2_cleaned = df_sheet_2_updated

df_sheet_1_cleaned = df_sheet_1.copy()
df_sheet_1_cleaned = df_sheet_1_cleaned.applymap(lambda x: x.strip() if isinstance(x, str) else x)

df_sheet_3_cleaned = df_sheet_3.copy()
df_sheet_3_cleaned = df_sheet_3_cleaned.applymap(lambda x: x.strip() if isinstance(x, str) else x)



mapping = {
    'A': '平旱地',
    'B': '梯田',
    'C': '山坡地',
    'D': '水浇地',
    'E': '普通大棚',
    'F': '智慧大棚'
}

df_sheet_3['种植地块'] = df_sheet_3['种植地块'].fillna(method='ffill')

def expand_row(row):
    crop_ids = str(row['作物编号']).split(',')
    crop_names = str(row['作物名称']).split(',')
    crop_types = str(row['作物类型']).split(',')
    areas = str(row['种植面积/亩']).split(',')
    length = max(len(crop_ids), len(crop_names), len(crop_types), len(areas))
    crop_ids += [None] * (length - len(crop_ids))
    crop_names += [None] * (length - len(crop_names))
    crop_types += [None] * (length - len(crop_types))
    areas += [None] * (length - len(areas))
    
    return [
        {
            '种植地块': row['种植地块'],
            '作物编号': crop_id,
            '作物名称': crop_name,
            '作物类型': crop_type,
            '种植面积/亩': area,
            '种植季次': row['种植季次']
        }
        for crop_id, crop_name, crop_type, area in zip(crop_ids, crop_names, crop_types, areas)
    ]
expanded_data = [item for sublist in df_sheet_3.apply(expand_row, axis=1) for item in sublist]

df_com = pd.DataFrame(expanded_data)
df_com['地块类型'] = df_com['种植地块'].str.extract(r'([A-Z])')[0].map(mapping)
path ='附件二去除合并单元格.xlsx'
df_com.to_excel(path, index=False)
df_sheet_4['地块类型'] = df_sheet_4['地块类型'].str.strip() 
df_sheet_4['种植季次'] = df_sheet_4['种植季次'].str.strip() 
df_com['地块类型'] = df_com['地块类型'].str.strip() 
df_com['种植季次'] = df_com['种植季次'].str.strip()  
df_sheet_4['地块类型'] = df_sheet_4['地块类型'].str.upper() 
df_com['地块类型'] = df_com['地块类型'].str.upper() 
df_sheet_4['种植季次'] = df_sheet_4['种植季次'].str.upper()  
df_com['种植季次'] = df_com['种植季次'].str.upper() 
df_com['作物编号'] = df_com['作物编号'].astype(str)
df_sheet_4['作物编号'] = df_sheet_4['作物编号'].astype(str)
df_com_filtered = df_com[['作物编号', '种植季次', '地块类型', '种植面积/亩']]
df_sheet_4_filtered = df_sheet_4[['作物编号', '种植季次', '地块类型', '亩产量/斤']]

df_merged = pd.merge(df_com_filtered, df_sheet_4_filtered, on=['作物编号', '种植季次', '地块类型'], how='left')

df_merged.at[77,'亩产量/斤'] =4000
df_merged.at[78,'亩产量/斤'] =4500
df_merged.at[81,'亩产量/斤'] =3600
df_merged.at[84,'亩产量/斤'] =3600
df_merged.at[73,'亩产量/斤'] =5500
df_merged.at[74,'亩产量/斤'] =6000


df_merged['种植面积/亩'] = pd.to_numeric(df_merged['种植面积/亩'], errors='coerce')
df_merged['亩产量/斤'] = pd.to_numeric(df_merged['亩产量/斤'], errors='coerce')

df_merged['总产量/斤'] = df_merged['种植面积/亩'] * df_merged['亩产量/斤']
df_merged['作物编号'] = df_merged['作物编号'].astype(int)


df_sales = df_merged.groupby('作物编号')['总产量/斤'].sum().reset_index()
df_sales = df_sales.sort_values(by='作物编号')
output_path = "sales.xlsx"
df_sales.to_excel(output_path, index=True)


sales = df_sales["总产量/斤"].tolist()
#print (len(sales))
#print(sales)




def calculate_midpoint(price_range):
    if isinstance(price_range, str):
        try:
            low, high = map(float, price_range.split('-'))
            return (low + high) / 2
        except ValueError:
            return np.nan
    else:
        return np.nan
df_sheet_4_cleaned['中点销售单价'] = df_sheet_4_cleaned['销售单价/(元/斤)'].apply(calculate_midpoint)

path='sheet_4.xlsx'
df_sheet_4_cleaned.to_excel(path) 








sales_a = sales[0:15]
sales_a.append(sales[0:15])
sales_a.append(sales[:34])
sales_a.append(sales[16:])
sales_a.append(sales[16:34])

sales_a = list(itertools.chain.from_iterable(
    item if isinstance(item, list) else [item] for item in sales_a
))

print(len(sales_a))




df_2024_2030 = df_sheet_4_cleaned[['作物编号', '作物名称', '亩产量/斤', '种植成本/(元/亩)', '中点销售单价']].copy()

print("合并后的数据（初始数据）：")
print(df_2024_2030.head())

years = range(2024, 2031)
growth_wheat_corn = np.random.uniform(0.05, 0.10, size=len(years))  # 小麦、玉米 5%-10% 增长
growth_other = np.random.uniform(-0.05, 0.05, size=len(years))  # 其他作物 -5%-5%
yield_variation = np.random.uniform(-0.10, 0.10, size=len(years))  # 亩产量波动 ±10%
cost_growth = 1.05 ** np.arange(len(years))  # 种植成本年增长5%
price_growth_vegetable = 1.05 ** np.arange(len(years))  # 蔬菜价格每年增长5%
price_decline_mushroom = np.random.uniform(0.95, 0.99, size=len(years))  # 食用菌价格下降1%-5%

df_randomized = pd.DataFrame(columns=['作物编号', '作物名称', '年份', '亩产量', '种植成本', '中点销售单价'])

for year, yield_var, cost_grow in zip(years, yield_variation, cost_growth):
    for idx, row in df_2024_2030.iterrows():
        new_row = {
            '作物编号': row['作物编号'],
            '作物名称': row['作物名称'],
            '年份': year,
            '亩产量': row['亩产量/斤'] * (1 + yield_var),
            '种植成本': row['种植成本/(元/亩)'] * cost_grow, 
        }
        
        if row['作物名称'] in ['豇豆''刀豆''芸豆''土豆''西红柿''茄子''菠菜''青椒''菜花''包菜''油麦菜'
                           '小青菜''黄瓜''生菜辣椒''空心菜''黄心菜''芹菜''大白菜''白萝卜''红萝卜']:
            new_row['中点销售单价'] = row['中点销售单价'] * price_growth_vegetable[year - 2024]
        elif row['作物名称'] in ['食用菌']:
            new_row['中点销售单价'] = row['中点销售单价'] * price_decline_mushroom[year - 2024]  
        else:
            new_row['中点销售单价'] = row['中点销售单价']

        df_randomized = pd.concat([df_randomized, pd.DataFrame([new_row])], ignore_index=True)

print("随机波动后的数据：")
print(df_randomized.head())

sales_a = sales[:107]  
sales_a = sales_a * len(years)  

def calculate_sales(row):
    year_index = row['年份'] - 2024  
    row_index = row.name % 107  
    if row['作物名称'] in ['小麦', '玉米']:
        return sales_a[row_index] * (1 + growth_wheat_corn[year_index])
    else:
        return sales_a[row_index] * (1 + growth_other[year_index])

df_randomized['sales'] = df_randomized.apply(calculate_sales, axis=1)
print(df_randomized.head())
df_randomized.to_excel("df_2024_2030_randomized_with_sales.xlsx", index=False)

df_2024_2030 = pd.merge(df_2024_2030, df_sheet_2[['作物编号', '作物类型']], on='作物编号', how='left')
df_2024_2030.head()


df_2024_2030.to_excel("df_2024_2030.xlsx")















df_land = pd.read_excel("附件1.xlsx", sheet_name="乡村的现有耕地")
crop_info = df_2024_2030[['作物编号', '作物名称', '作物类型', '中点销售单价', '亩产量/斤']]

legume_crops = crop_info[crop_info['作物类型'].isin(['粮食（豆类）', '蔬菜（豆类）'])]['作物编号'].unique()

crop_info['作物类型'] = crop_info['作物类型'].replace({
    '粮食（豆类）': '粮食',
    '蔬菜（豆类）': '蔬菜'
})



land_history = {land_id: {} for land_id in df_land['地块名称'].unique()}

def adjust_area_cost_for_complement_and_substitute(crop_id, sales_price_change, crop_areas, cost_change, crop_info):
    crop_type_j = crop_info[crop_info['作物编号'] == crop_id]['作物类型'].values[0]

    for related_crop_id in crop_info['作物编号'].unique():
        if related_crop_id == crop_id:
            continue  
        
        crop_type_k = crop_info[crop_info['作物编号'] == related_crop_id]['作物类型'].values[0]
        
        related_idx = crop_info.loc[crop_info['作物编号'] == related_crop_id].index
        
        if related_idx.empty or related_idx[0] >= len(crop_areas):
            continue  
        
        if crop_type_j == crop_type_k:  # 替代品
            if sales_price_change > 0:  # 作物j的价格上升
                crop_areas[related_idx[0]] -= np.random.uniform(0.05, 0.10) * crop_areas[related_idx[0]] 
                cost_change[related_idx[0]] += np.random.uniform(0.05, 0.10) 
            else:  # 作物j的价格下降
                crop_areas[related_idx[0]] += np.random.uniform(0.05, 0.10) * crop_areas[related_idx[0]]
                cost_change[related_idx[0]] -= np.random.uniform(0.05, 0.10)  # 成本减少
        else:  # 互补品
            if sales_price_change > 0: 
                crop_areas[related_idx[0]] += np.random.uniform(0.05, 0.10) * crop_areas[related_idx[0]]  
                cost_change[related_idx[0]] -= np.random.uniform(0.05, 0.10) 
            else: 
                crop_areas[related_idx[0]] -= np.random.uniform(0.05, 0.10) * crop_areas[related_idx[0]] 
                cost_change[related_idx[0]] += np.random.uniform(0.05, 0.10) 
def single_iteration(years, df_land, df_2024_2030, df_randomized, legume_crops, land_history):
    total_revenue = 0
    current_solution = {}

    for year in years:
        for land_id in df_land['地块名称'].unique():
            available_area = df_land[df_land['地块名称'] == land_id]['地块面积/亩'].values[0]
            crop_areas = np.zeros(len(df_2024_2030['作物编号'].unique()))
            cost_change = np.zeros(len(df_2024_2030['作物编号'].unique()))  
            remaining_area = available_area

            for i, crop_id in enumerate(df_2024_2030['作物编号'].unique()):
                if year > 2024 and land_id in land_history and land_history[land_id].get(year - 1) == crop_id:
                    continue  

                crop_areas[i] = np.random.uniform(0, remaining_area)
                remaining_area -= crop_areas[i]
                if remaining_area <= 0:
                    break

            # 确保三年内至少种植一次豆类作物
            if year % 3 == 0 and not any(df_2024_2030['作物编号'].isin(legume_crops)):
                legume_idx = np.random.choice(np.where(df_2024_2030['作物编号'].isin(legume_crops))[0])
                crop_areas[legume_idx] = max(5, np.random.uniform(0, available_area))  # 至少种植5亩

            # 确保每种作物的最小种植面积
            crop_areas = np.maximum(crop_areas, 5)

            # 根据地块类型适应性分配作物
            land_type = df_land[df_land['地块名称'] == land_id]['地块类型'].values[0]
            for i, crop_id in enumerate(df_2024_2030['作物编号'].unique()):
                crop_type = crop_info[crop_info['作物编号'] == crop_id]['作物类型'].values[0]
                if land_type == '平旱地' and crop_type != '粮食':
                    crop_areas[i] = 0
                elif land_type == '水浇地' and crop_type not in ['水稻', '蔬菜']:
                    crop_areas[i] = 0
                elif land_type == '普通大棚' and crop_type not in ['蔬菜', '食用菌']:
                    crop_areas[i] = 0
                elif land_type == '智慧大棚' and crop_type != '蔬菜':
                    crop_areas[i] = 0

            # 计算每个作物的总产量并计算收益
            for crop_id, area in zip(df_2024_2030['作物编号'].unique(), crop_areas):
                yield_per_mu = df_2024_2030[df_2024_2030['作物编号'] == crop_id]['亩产量/斤'].values[0]
                total_yield = area * yield_per_mu
                sales = df_randomized[(df_randomized['作物编号'] == crop_id) & (df_randomized['年份'] == year)]['sales'].values[0]
                sales_price = df_2024_2030[df_2024_2030['作物编号'] == crop_id]['中点销售单价'].values[0]

                # 计算作物j的销售价格波动
                sales_price_change = np.random.uniform(-0.05, 0.05)  
                adjust_area_cost_for_complement_and_substitute(crop_id, sales_price_change, crop_areas, cost_change, crop_info)

                # 产量与销售量的匹配
                if total_yield <= sales:
                    total_revenue += total_yield * sales_price
                else:
                    excess = total_yield - sales
                    total_revenue += sales * sales_price + excess * sales_price * 0.5  # 超出部分按50%出售
                current_solution[(crop_id, land_id, year)] = area

            land_history[land_id][year] = np.argmax(crop_areas) 

    return total_revenue

num_iterations = 100 
years = range(2024, 2031)  

results = Parallel(n_jobs=-1)(delayed(single_iteration)(years, df_land, df_2024_2030, df_randomized, legume_crops, land_history) for _ in range(num_iterations))
best_revenue = max(results)
print("最优方案的总收益：", best_revenue)

