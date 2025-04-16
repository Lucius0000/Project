# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 16:42:26 2024

@author: Lucius
"""

import pandas as pd
import numpy as np
import sympy as sp
from scipy.optimize import linprog,minimize
import re




attach_1=r"C:\Users\Lucius\Desktop\CUMCM2024Problems\C题\附件1.xlsx"
attach_2=r"C:\Users\Lucius\Desktop\CUMCM2024Problems\C题\附件2.xlsx"
sheet_1="乡村的现有耕地"
sheet_2="乡村种植的农作物"
sheet_3="2023年的农作物种植情况"
sheet_4="2023年统计的相关数据"
df_1 = pd.read_excel(attach_1, sheet_name=sheet_1)
df_2 = pd.read_excel(attach_1, sheet_name=sheet_2)
df_3 = pd.read_excel(attach_2, sheet_name=sheet_3)
df_4 = pd.read_excel(attach_2, sheet_name=sheet_4)
df_4=df_4.drop(index=107).drop(index=108).drop(index=109)


# 统一列名，去除列名中的空格
df_1.columns = df_1.columns.str.strip()
df_2.columns = df_2.columns.str.strip()
df_3.columns = df_3.columns.str.strip()
df_4.columns = df_4.columns.str.strip()

# 去除数据中的空格
df_1 = df_1.applymap(lambda x: x.strip() if isinstance(x, str) else x)
df_2 = df_2.applymap(lambda x: x.strip() if isinstance(x, str) else x)
df_3 = df_3.applymap(lambda x: x.strip() if isinstance(x, str) else x)
df_4 = df_4.applymap(lambda x: x.strip() if isinstance(x, str) else x)







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
df_3['种植地块'] = df_3['种植地块'].fillna(method='ffill')

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
expanded_data = [item for sublist in df_3.apply(expand_row, axis=1) for item in sublist]

# 将扩展后的数据转换为 DataFrame
df_com = pd.DataFrame(expanded_data)

# 添加地块类型列
df_com['地块类型'] = df_com['种植地块'].str.extract(r'([A-Z])')[0].map(mapping)

# 保存调整后的数据
path ='附件二去除合并单元格.xlsx'
df_com.to_excel(path, index=False)

# 处理 df_4 中的空格
df_4['地块类型'] = df_4['地块类型'].str.strip() 
df_4['种植季次'] = df_4['种植季次'].str.strip() 

# 处理 df_com 中的空格
df_com['地块类型'] = df_com['地块类型'].str.strip() 
df_com['种植季次'] = df_com['种植季次'].str.strip()  

# 统一大小写
df_4['地块类型'] = df_4['地块类型'].str.upper() 
df_com['地块类型'] = df_com['地块类型'].str.upper() 
df_4['种植季次'] = df_4['种植季次'].str.upper()  
df_com['种植季次'] = df_com['种植季次'].str.upper() 

# 确保数据类型一致
df_com['作物编号'] = df_com['作物编号'].astype(str)
df_4['作物编号'] = df_4['作物编号'].astype(str)

# 提取用于合并的列
df_com_filtered = df_com[['作物编号', '种植季次', '地块类型', '种植面积/亩']]
df_4_filtered = df_4[['作物编号', '种植季次', '地块类型', '亩产量/斤']]

df_merged = pd.merge(df_com_filtered, df_4_filtered, on=['作物编号', '种植季次', '地块类型'], how='left')


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





df_3['作物编号'] = df_3['作物编号'].astype(int)
df_4['作物编号'] = df_4['作物编号'].astype(int)






#计算单亩产出
df_c = df_4[['销售单价/(元/斤)', '亩产量/斤', '作物编号', '地块类型','种植季次']]
# 确保销售单价是字符串类型
df_c['销售单价/(元/斤)'] = df_c['销售单价/(元/斤)'].astype(str)

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

# 计算每一行的单亩产出均值
df_c['中点销售单价'] = df_c['销售单价/(元/斤)'].apply(calculate_midpoint)
df_c['单亩产出均值'] = df_c['中点销售单价'] * df_c['亩产量/斤'] 

# 合并地块类型和种植季次列为一个新的列
df_c['地块类型_种植季次'] = df_c['地块类型'] + df_c['种植季次']


# 创建dataframe：行索引为“作物编号”，列索引为“地块类型”，值为“单亩产出均值”
pivot_table = df_c.pivot_table(
    index=['地块类型_种植季次'],
    columns='作物编号',
    values='单亩产出均值',
    aggfunc='mean'
)
data_copy = pivot_table.loc ["普通大棚第一季",:]
pivot_table.loc["智慧大棚第一季"] = data_copy
pivot_table.fillna(0, inplace=True)


#print(pivot_table)
output_path = 'output_pivot_table.xlsx'
pivot_table.to_excel(output_path)









# 创建可以矩阵相乘的单亩产出矩阵
column1 = list(range(1, 42))
B = pd.DataFrame(columns=column1)

# 定义函数以重复数据并将其添加到 DataFrame 中
def add_repeated_rows(data_copy, repeat_count):
    data_repeated = pd.DataFrame([data_copy] * repeat_count)
    return pd.concat([B, data_repeated], ignore_index=True)

data_copy = pivot_table.loc["平旱地单季", :]
B = add_repeated_rows(data_copy, 6)

data_copy = pivot_table.loc["梯田单季", :]
B = add_repeated_rows(data_copy, 14)

data_copy = pivot_table.loc["山坡地单季", :]
B = add_repeated_rows(data_copy, 6)

data_copy = pivot_table.loc["水浇地单季", :]
B = add_repeated_rows(data_copy, 8)

data_copy = pivot_table.loc["水浇地第一季", :]
B = add_repeated_rows(data_copy, 8)

data_copy = pivot_table.loc["水浇地第二季", :]
B = add_repeated_rows(data_copy, 8)

data_copy = pivot_table.loc["普通大棚第一季", :]
B = add_repeated_rows(data_copy, 16)

data_copy = pivot_table.loc["普通大棚第二季", :]
B = add_repeated_rows(data_copy, 16)

data_copy = pivot_table.loc["智慧大棚第一季", :]
B = add_repeated_rows(data_copy, 4)

data_copy = pivot_table.loc["智慧大棚第二季", :]
B = add_repeated_rows(data_copy, 4)

#print(B)
output_path = 'B.xlsx'
B.to_excel(output_path, index=True)









# 创建符号变量矩阵
rows, cols = 41, 90
variables = [sp.Symbol(f'x{i+1}') for i in range(rows * cols)]
A = pd.DataFrame([variables[i*cols:(i+1)*cols] for i in range(rows)])

# 先将整个 DataFrame 赋值为0
A[:] = 0
# 确保区域数据的形状一致
def set_data(start_row, end_row, start_col, end_col):
    data = [[variables[i*cols + j] for j in range(start_col, end_col)] for i in range(start_row, end_row)]
    if np.array(data).shape == A.loc[start_row:end_row-1, start_col:end_col-1].shape:
        A.loc[start_row:end_row-1, start_col:end_col-1] = data
    else:
        print(f"Shape mismatch for region ({start_row}:{end_row}, {start_col}:{end_col}):", np.array(data).shape, A.loc[start_row:end_row-1, start_col:end_col-1].shape)

# 设置不同区域的数据
set_data(0, 15, 0, 26)       # 0-14行交0-25列
set_data(15, 16, 26, 34)     # 15行交26-33列
set_data(16, 34, 34, 42)     # 16-33行交34-41列
set_data(34, 37, 42, 50)     # 34-36行交42-49列
set_data(16, 34, 50, 66)     # 16-33行交50-64列
set_data(37, 41, 66, 82)     # 37-40行交66-80列
set_data(16, 34, 82, 86)     # 16-33行交82-84列
set_data(16, 34, 86, 90)     # 16-33行交86-88列

# 输出到 Excel 文件
output_path = 'A.xlsx'
A.to_excel(output_path, index=True)
#print(A)









#创建成本矩阵
df_d=df_4[['作物编号','地块类型','种植季次','种植成本/(元/亩)']]
df_d['地块类型_种植季次'] = df_d['地块类型'] + df_d['种植季次']       
cost = df_d.pivot_table(
    index='地块类型_种植季次',
    columns='作物编号',
    values='种植成本/(元/亩)')
cost.fillna(0, inplace=True)
data_copy = cost.loc ["普通大棚第一季",:]
cost.loc["智慧大棚第一季"] = data_copy
#print(cost)

column1 = list(range(1, 42))
D = pd.DataFrame(columns=column1)

# 定义函数以重复数据并将其添加到 DataFrame 中
def add_repeated_rows(data_copy, repeat_count):
    data_repeated = pd.DataFrame([data_copy] * repeat_count)
    return pd.concat([D, data_repeated], ignore_index=True)

data_copy = cost.loc["平旱地单季", :]
D = add_repeated_rows(data_copy, 6)

data_copy = cost.loc["梯田单季", :]
D = add_repeated_rows(data_copy, 14)

data_copy = cost.loc["山坡地单季", :]
D = add_repeated_rows(data_copy, 6)

data_copy = cost.loc["水浇地单季", :]
D = add_repeated_rows(data_copy, 8)

data_copy =cost.loc["水浇地第一季", :]
D = add_repeated_rows(data_copy, 8)

data_copy = cost.loc["水浇地第二季", :]
D = add_repeated_rows(data_copy, 8)

data_copy = cost.loc["普通大棚第一季", :]
D = add_repeated_rows(data_copy, 16)

data_copy =cost.loc["普通大棚第二季", :]
D = add_repeated_rows(data_copy, 16)

data_copy =cost.loc["智慧大棚第一季", :]
D = add_repeated_rows(data_copy, 4)

data_copy = cost.loc["智慧大棚第二季", :]
D = add_repeated_rows(data_copy, 4)

#print(D)
output_path = 'D.xlsx'
D.to_excel(output_path, index=False)









#创建地块面积行向量
df_e = df_1[["地块面积/亩"]]
# 转换为列表
list1 = df_e['地块面积/亩'].values.flatten().tolist()

list2 = list1[:34]
list2.append(list1[26:34])
list2.append(list1[26:34])
list2.append(list1[34:50])
list2.append(list1[34:50])
list2.append(list1[50:])
list2.append(list1[50:])

s1=[item for sublist in list2 for item in (sublist if isinstance(sublist, list) else [sublist])]

#print (s1)









#求利润
#先求销售量
#在滞销的情况下，不允许产量超过预期销售量
#41*90与90*41相乘，得到41*41方阵
#取对角线上所有元素，每个元素，如sum1的（1，1），即表示小麦，7年，所有土地，总收入
#需要留意的是，输出结果sum1的排列并不是严格地以矩阵乘法的顺序，而是按x的下标排列，1，10，100……2,20……
#仅仅是一年的数据，要用矩阵串联*7
#因为仅仅需要乘积矩阵的对角线，故尝试使用np.einsum()函数减轻运算量，但是结果有误可能是理解不到位吧，故放弃
#矩阵a不是简单水平堆叠，因为不能有7个x1，故更改变量名复制7次；b，d是纯数据，直接垂直堆叠

# 复制并替换符号变量的前缀
prefixes = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
matrices = []

for prefix in prefixes:
    # 替换 'x' 为新的前缀
    new_matrix = A.applymap(lambda val: sp.Symbol(str(val).replace('x', prefix)) if isinstance(val, sp.Symbol) else val)
    matrices.append(new_matrix)

# 合并7个矩阵，横向堆叠
E = pd.concat(matrices, axis=1, ignore_index=True)

output_path = 'E.xlsx'
E.to_excel(output_path, index=True)



a=E.to_numpy()
b=B.to_numpy()
d=D.to_numpy()

b = np.vstack([b] * 7)
d = np.vstack([d] * 7)

c = a @ b
sum1 = np.diag(c)
sum1 = np.sum(sum1)

check1 = open ("check.txt","w")
check1.write(f"{sum1}")
check1.close()

#print (sum1)

e = a @ d
sum2 = np.diag(e)
sum2 = np.sum(sum2)

check2 = open ("check2.txt","w")
check2.write(f"{sum2}")
check2.close()

w = sum1 - sum2
check3 = open("check3.txt","w")
check3.write(f"{w}")
check3.close()


'''
#校验
a=pd.DataFrame(b)
path = "exam1.xlsx"
a.to_excel(path,index=True)
print(a)
'''






#设置限制条件
'''
# 解析 w 表达式
def parse_w_expression(w_expression):
    # 确保 w_expression 是字符串
    if not isinstance(w_expression, str):
        w_expression = str(w_expression)
    
    terms = w_expression.split('+')
    w_dict = {}
    for term in terms:
        coeff, var = term.split('*')
        coeff = float(coeff.strip())
        var = var.strip()
        w_dict[var] = coeff
    return w_dict


# 将 w 表达式解析为字典
w_dict = parse_w_expression(w)

# 提取 w 中的所有变量
variables = list(w_dict.keys())
'''

# 初始解
w0 = np.ones(E.shape[1]) * 6  # 初始化为所有变量等于6

# 定义目标函数
def objective(w):
    return -np.sum(w)  # 假设目标是最大化 w 的和，因此取负号最小化

# 定义约束条件
def constraints(w):
    constraints_list = []

    # 约束1：w 中所有非零元素的数量必须大于等于6
    constraints_list.append({'type': 'ineq', 'fun': lambda w: np.count_nonzero(w) - 6})

    # 约束2：E 矩阵每列的和小于 s1 中对应的元素
    for j in range(E.shape[1]):
        s1_index = j % 90  # 循环使用 s1 的元素
        constraints_list.append({'type': 'ineq', 'fun': lambda w, j=j, s1_index=s1_index: s1[s1_index] - np.sum(E.iloc[:, j] * w)})

    # 约束3：E 矩阵每行分段和小于 sales 的元素
    for i in range(E.shape[0]):
        for seg in range(0, 630, 90):
            sales_index = i % 90
            constraints_list.append({'type': 'ineq', 'fun': lambda w, i=i, seg=seg, sales_index=sales_index: sales[sales_index] - np.sum(E.iloc[i, seg:seg + 90] * w)})

    # 约束4：相邻变量下标相同、字母相邻的变量限制
    for subscript in range(1, 3682):  # 根据实际变量的数量调整范围
        for letter in 'abcdefg':  # 遍历相邻的字母对 a-b, b-c, ..., f-g
            if letter != 'g':  # 因为 g 没有相邻的字母
                var1 = f"{letter}{subscript}"
                var2 = f"{chr(ord(letter) + 1)}{subscript}"
                # 添加约束：相邻的两个变量中至少有一个为 0
                constraints_list.append({'type': 'ineq', 'fun': lambda w, var1=var1, var2=var2: 1 - w[variables.index(var1)] - w[variables.index(var2)]})
'''
    # 约束5：特定行列求和后大于 s1
    for j in range(E.shape[1]):
        s1_index = j % 90
        E_subset = E.iloc[:, j]  # 假设 E 具有适当的维度
        constraint_value = np.sum(E_subset * w)
        constraints_list.append({'type': 'ineq', 'fun': lambda w, s1_index=s1_index, constraint_value=constraint_value: s1[s1_index] - constraint_value})
'''
# 优化求解
res = minimize(objective, w0, method='SLSQP', constraints=constraints(w0), options={'disp': True})

result = res.x
resul_profit = res.fun
print("最大化后的 w：", -res.fun)
print(result)
file = open("result1.1","w")
file.writestr((result))




#生成完整的参数名列表（7个block，每个block从1到3682）
parameter_names = []
for block_prefix in ['a', 'b', 'c', 'd', 'e', 'f', 'g']:
    for i in range(1, 3683):  # 生成 a1 到 g3682
        parameter_names.append(f'{block_prefix}{i}')

#对参数名按照数字自然顺序进行排序
def natural_key(s):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]

parameter_names_sorted = sorted(parameter_names, key=natural_key)

#初始化参数值列表，将不存在的变量值设置为0
param_values = [0] * len(parameter_names_sorted)

#将 result 按顺序填入 param_values 中，覆盖默认的 0
for i in range(len(result)):
    param_values[i] = result[i]

df_result = pd.DataFrame({
    'Parameter': parameter_names_sorted,
    'Value': param_values
})


output_file = 'optimal_params_natural_sorted.xlsx'
df_result.to_excel(output_file, index=False)

print(f"优化结果已保存到 {output_file}")




'''
# 输出优化后的 w 值
w_optimized = res.x
print("最大化后的 w：", -res.fun)


# 创建一个字典来保存优化后的变量值
optimized_values = {var: w_optimized[i] for i, var in enumerate(variables)}

# 更新 E 矩阵，将优化后的 w 结果替代进去
for col in E.columns:
    if col in optimized_values:
        E[col] = optimized_values[col]
        

# 输出矩阵 E 到 Excel 文件
output_path = 'E_optimized.xlsx'
E.to_excel(output_path, index=True)

print(f"已保存优化后的 E 矩阵到: {output_path}")
'''
































