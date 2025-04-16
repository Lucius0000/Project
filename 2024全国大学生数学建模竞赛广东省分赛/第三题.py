
import pandas as pd
import numpy as np
import itertools



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


# 清洗 df_sheet_4_cleaned 中的地块类型信息，确保每个作物的种植耕地信息正确
df_sheet_4_cleaned['地块类型'] = df_sheet_4_cleaned['地块类型'].str.strip()

# 创建作物编号和种植耕地的对应关系
crop_land_mapping = df_sheet_4_cleaned[['作物编号', '地块类型']].dropna().drop_duplicates()

# 根据作物编号进行分组，并将多个地块类型进行合并为一个字符串
crop_land_mapping_grouped = crop_land_mapping.groupby('作物编号')['地块类型'].apply(lambda x: ','.join(x.unique())).reset_index()

# 将作物编号对应的种植耕地替换到 sheet2 的 "种植耕地" 列中
df_sheet_2_updated = df_sheet_2.copy()

# 根据作物编号进行映射
df_sheet_2_updated = df_sheet_2_updated.merge(crop_land_mapping_grouped, on='作物编号', how='left')

# 使用更新后的地块类型信息
df_sheet_2_updated['种植耕地'] = df_sheet_2_updated['地块类型']

# 删除不必要的地块类型列
df_sheet_2_updated = df_sheet_2_updated.drop(columns=['地块类型'])

# 进一步清洗数据，去除多余的空格和换行符，处理种植耕地列中的异常字符
df_sheet_2_updated['种植耕地'] = df_sheet_2_updated['种植耕地'].str.replace(r'\s+', '', regex=True)

# 清洗说明列中的空白符
df_sheet_2_updated['说明'] = df_sheet_2_updated['说明'].str.strip()
df_sheet_2_cleaned = df_sheet_2_updated

df_sheet_1_cleaned = df_sheet_1.copy()
df_sheet_1_cleaned = df_sheet_1_cleaned.applymap(lambda x: x.strip() if isinstance(x, str) else x)

df_sheet_3_cleaned = df_sheet_3.copy()
df_sheet_3_cleaned = df_sheet_3_cleaned.applymap(lambda x: x.strip() if isinstance(x, str) else x)








#求预期销售量
# 定义地块类型映射字典
mapping = {
    'A': '平旱地',
    'B': '梯田',
    'C': '山坡地',
    'D': '水浇地',
    'E': '普通大棚',
    'F': '智慧大棚'
}

# 填充缺失的“种植地块”信息
df_sheet_3['种植地块'] = df_sheet_3['种植地块'].fillna(method='ffill')

# 扩展数据的函数
def expand_row(row):
    # 拆解作物信息
    crop_ids = str(row['作物编号']).split(',')
    crop_names = str(row['作物名称']).split(',')
    crop_types = str(row['作物类型']).split(',')
    areas = str(row['种植面积/亩']).split(',')
    
    # 确保长度一致
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

# 扩展所有行
expanded_data = [item for sublist in df_sheet_3.apply(expand_row, axis=1) for item in sublist]

# 将扩展后的数据转换为 DataFrame
df_com = pd.DataFrame(expanded_data)

# 添加地块类型列
df_com['地块类型'] = df_com['种植地块'].str.extract(r'([A-Z])')[0].map(mapping)

# 保存调整后的数据
path ='附件二去除合并单元格.xlsx'
df_com.to_excel(path, index=False)

# 处理 df_sheet_4 中的空格
df_sheet_4['地块类型'] = df_sheet_4['地块类型'].str.strip() 
df_sheet_4['种植季次'] = df_sheet_4['种植季次'].str.strip() 

# 处理 df_com 中的空格
df_com['地块类型'] = df_com['地块类型'].str.strip() 
df_com['种植季次'] = df_com['种植季次'].str.strip()  

# 统一大小写
df_sheet_4['地块类型'] = df_sheet_4['地块类型'].str.upper() 
df_com['地块类型'] = df_com['地块类型'].str.upper() 
df_sheet_4['种植季次'] = df_sheet_4['种植季次'].str.upper()  
df_com['种植季次'] = df_com['种植季次'].str.upper() 

# 确保数据类型一致
df_com['作物编号'] = df_com['作物编号'].astype(str)
df_sheet_4['作物编号'] = df_sheet_4['作物编号'].astype(str)

# 提取用于合并的列
df_com_filtered = df_com[['作物编号', '种植季次', '地块类型', '种植面积/亩']]
df_sheet_4_filtered = df_sheet_4[['作物编号', '种植季次', '地块类型', '亩产量/斤']]

df_merged = pd.merge(df_com_filtered, df_sheet_4_filtered, on=['作物编号', '种植季次', '地块类型'], how='left')


#智慧大棚第一季缺失
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








# 函数：从区间计算中点
def calculate_midpoint(price_range):
    if isinstance(price_range, str):
        try:
            low, high = map(float, price_range.split('-'))
            return (low + high) / 2
        except ValueError:
            # 如果分割失败或格式不正确，返回NaN
            return np.nan
    else:
        # 如果不是字符串类型，返回NaN
        return np.nan
df_sheet_4_cleaned['中点销售单价'] = df_sheet_4_cleaned['销售单价/(元/斤)'].apply(calculate_midpoint)

path='sheet_4.xlsx'
df_sheet_4_cleaned.to_excel(path) 








sales_a = sales[0:15]
sales_a.append(sales[0:15])
sales_a.append(sales[:34])
sales_a.append(sales[16:])
sales_a.append(sales[16:34])

#展平嵌套列表
sales_a = list(itertools.chain.from_iterable(
    item if isinstance(item, list) else [item] for item in sales_a
))

print(len(sales_a))




df_2024_2030 = df_sheet_4_cleaned[['作物编号', '作物名称', '亩产量/斤', '种植成本/(元/亩)', '中点销售单价']].copy()

# 打印合并后的数据检查
print("合并后的数据（初始数据）：")
print(df_2024_2030.head())

# 定义年份和随机波动
years = range(2024, 2031)
growth_wheat_corn = np.random.uniform(0.05, 0.10, size=len(years))  # 小麦、玉米 5%-10% 增长
growth_other = np.random.uniform(-0.05, 0.05, size=len(years))  # 其他作物 -5%-5%
yield_variation = np.random.uniform(-0.10, 0.10, size=len(years))  # 亩产量波动 ±10%
cost_growth = 1.05 ** np.arange(len(years))  # 种植成本年增长5%
price_growth_vegetable = 1.05 ** np.arange(len(years))  # 蔬菜价格每年增长5%
price_decline_mushroom = np.random.uniform(0.95, 0.99, size=len(years))  # 食用菌价格下降1%-5%

# 初始化存储随机波动后的数据
df_randomized = pd.DataFrame(columns=['作物编号', '作物名称', '年份', '亩产量', '种植成本', '中点销售单价'])

# 对每个年份逐步应用波动生成新的数据
for year, yield_var, cost_grow in zip(years, yield_variation, cost_growth):
    for idx, row in df_2024_2030.iterrows():
        new_row = {
            '作物编号': row['作物编号'],
            '作物名称': row['作物名称'],
            '年份': year,
            '亩产量': row['亩产量/斤'] * (1 + yield_var),  # 亩产量波动
            '种植成本': row['种植成本/(元/亩)'] * cost_grow,  # 种植成本年增长5%
        }
        
        # 销售单价按不同作物类别进行调整
        if row['作物名称'] in ['豇豆''刀豆''芸豆''土豆''西红柿''茄子''菠菜''青椒''菜花''包菜''油麦菜'
                           '小青菜''黄瓜''生菜辣椒''空心菜''黄心菜''芹菜''大白菜''白萝卜''红萝卜']:
            new_row['中点销售单价'] = row['中点销售单价'] * price_growth_vegetable[year - 2024]  # 蔬菜类价格每年增长5%
        elif row['作物名称'] in ['食用菌']:
            new_row['中点销售单价'] = row['中点销售单价'] * price_decline_mushroom[year - 2024]  # 食用菌价格下降
        else:
            new_row['中点销售单价'] = row['中点销售单价']

        # 将生成的新数据行添加到 df_randomized 中
        df_randomized = pd.concat([df_randomized, pd.DataFrame([new_row])], ignore_index=True)

# 打印随机波动后的数据检查
print("随机波动后的数据：")
print(df_randomized.head())

# 生成 sales_a 并确保每年对应 107 行
sales_a = sales[:107]  # 每年 107 行
sales_a = sales_a * len(years)  # 复制足够的行数

# 使用 apply 函数来计算 sales 列
def calculate_sales(row):
    year_index = row['年份'] - 2024  # 年份从 2024 开始
    row_index = row.name % 107  # 使用 mod 107 确保每年 107 行数据
    if row['作物名称'] in ['小麦', '玉米']:
        return sales_a[row_index] * (1 + growth_wheat_corn[year_index])
    else:
        return sales_a[row_index] * (1 + growth_other[year_index])

# 应用 calculate_sales 函数
df_randomized['sales'] = df_randomized.apply(calculate_sales, axis=1)

# 查看结果
print(df_randomized.head())

# 保存最终结果
df_randomized.to_excel("df_2024_2030_randomized_with_sales.xlsx", index=False)


# 将 df_2024_2030 与 df_sheet_2 基于 '作物编号' 进行合并，补充 '作物类型'
df_2024_2030 = pd.merge(df_2024_2030, df_sheet_2[['作物编号', '作物类型']], on='作物编号', how='left')
df_2024_2030.head()


df_2024_2030.to_excel("df_2024_2030.xlsx")

















# 读取数据
df_land = pd.read_excel("附件1.xlsx", sheet_name="乡村的现有耕地")
df_2024_2030 = pd.read_excel("df_2024_2030.xlsx")

# 获取作物的种类信息
crop_info = df_2024_2030[['作物编号', '作物名称', '作物类型', '中点销售单价', '亩产量/斤']]

# 修改豆类作物的分类，确保豆类作物按规定视作相同类型
crop_info['作物类型'] = crop_info['作物类型'].replace({
    '粮食（豆类）': '粮食',
    '蔬菜（豆类）': '蔬菜'
})

path="crop.xlsx"
crop_info.to_excel(path)

legume_crops = crop_info[crop_info['作物类型'].isin(['粮食', '蔬菜'])]['作物编号'].unique()


# 约束和随机波动相关参数
num_iterations = 100  # 迭代次数
years = range(2024, 2031)
best_solution = None
best_revenue = -np.inf  # 初始化最大收益


# 定义互补品和替代品调整规则
def adjust_area_cost_for_complement_and_substitute(crop_id, sales_price_change, crop_areas, cost_change, crop_info):
    crop_type_j = crop_info[crop_info['作物编号'] == crop_id]['作物类型'].values[0]

    for related_crop_id in crop_info['作物编号'].unique():
        if related_crop_id == crop_id:
            continue  # 跳过自身
        
        crop_type_k = crop_info[crop_info['作物编号'] == related_crop_id]['作物类型'].values[0]
        
        # 使用loc[]获取作物编号对应的索引，确保索引在有效范围内
        related_idx = crop_info.loc[crop_info['作物编号'] == related_crop_id].index
        
        # 检查索引是否超出数组范围
        if related_idx.empty or related_idx[0] >= len(crop_areas):
            continue  # 跳过不存在的索引
        
        # 判断作物之间的关系
        if crop_type_j == crop_type_k:  # 替代品
            if sales_price_change > 0:  # 作物j的价格上升
                crop_areas[related_idx[0]] -= np.random.uniform(0.05, 0.10) * crop_areas[related_idx[0]]  # 种植面积减少
                cost_change[related_idx[0]] += np.random.uniform(0.05, 0.10)  # 成本增加
            else:  # 作物j的价格下降
                crop_areas[related_idx[0]] += np.random.uniform(0.05, 0.10) * crop_areas[related_idx[0]]  # 种植面积增加
                cost_change[related_idx[0]] -= np.random.uniform(0.05, 0.10)  # 成本减少
        else:  # 互补品
            if sales_price_change > 0:  # 作物j的价格上升
                crop_areas[related_idx[0]] += np.random.uniform(0.05, 0.10) * crop_areas[related_idx[0]]  # 种植面积增加
                cost_change[related_idx[0]] -= np.random.uniform(0.05, 0.10)  # 成本减少
            else:  # 作物j的价格下降
                crop_areas[related_idx[0]] -= np.random.uniform(0.05, 0.10) * crop_areas[related_idx[0]]  # 种植面积减少
                cost_change[related_idx[0]] += np.random.uniform(0.05, 0.10)  # 成本增加
                

# 随机搜索
for iteration in range(num_iterations):
    total_revenue = 0  # 初始化总收益
    current_solution = {}

    for year in years:
        for land_id in df_land['地块名称'].unique():
            available_area = df_land[df_land['地块名称'] == land_id]['地块面积/亩'].values[0]
            crop_areas = np.zeros(len(df_2024_2030['作物编号'].unique()))
            cost_change = np.zeros(len(df_2024_2030['作物编号'].unique()))  # 用于存储作物成本的波动情况
            remaining_area = available_area

            # 随机分配种植面积
            for i, crop_id in enumerate(df_2024_2030['作物编号'].unique()):
                crop_areas[i] = np.random.uniform(0, remaining_area)
                remaining_area -= crop_areas[i]
                if remaining_area <= 0:
                    break

            # 确保三年内至少种植一次豆类作物
            if year % 3 == 0 and not any(df_2024_2030['作物编号'].isin(legume_crops)):
                legume_idx = np.random.choice(np.where(df_2024_2030['作物编号'].isin(legume_crops))[0])
                crop_areas[legume_idx] = max(5, np.random.uniform(0, available_area))  # 至少种植5亩

            # 确保同一地块不能连续种植相同作物
            if year > 2024 and (crop_id, land_id, year-1) in current_solution:
                crop_areas[i] = 0  # 防止连续种植相同作物

            # 确保每种作物的最小种植面积（5亩）
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
                sales_price_change = np.random.uniform(-0.05, 0.05)  # 假设销售价格变化
                adjust_area_cost_for_complement_and_substitute(crop_id, sales_price_change, crop_areas, cost_change, crop_info)

                # 产量与销售量的匹配
                if total_yield <= sales:
                    total_revenue += total_yield * sales_price
                else:
                    excess = total_yield - sales
                    total_revenue += sales * sales_price + excess * sales_price * 0.5  # 超出部分按50%出售

                # 存储当前的种植面积
                current_solution[(crop_id, land_id, year)] = area

        # 更新最优方案
        if total_revenue > best_revenue:
            best_revenue = total_revenue
            best_solution = current_solution

# 输出最优结果
print("最优方案的总收益：", best_revenue)
