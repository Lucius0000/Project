import pandas as pd

# 读取 Excel 文件
file_path = r"C:\Users\Lucius\Desktop\data_changed\program.xlsx"  # 替换为你的文件路径
df = pd.read_excel(file_path)

# 查看读取的表格内容
print("Original DataFrame:")
print(df)

# 将所有的 "•" 替换为 0
df.replace('•', 0, inplace=True)

# 将 NaN 替换为 0（确保修改会生效）
df.fillna(0, inplace=True)

df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

# 将 Sport 列和年份列按 Sport 分组，并对其他列进行求和
df_grouped = df.groupby('Sport').sum()


# 查看合并后的结果
print("Grouped DataFrame (Sum of Values):")
print(df_grouped)

# 将数据转换为长格式
df_long = df_grouped.reset_index().melt(id_vars=["Sport"], var_name="Year", value_name="event_count")

# 将 event_count 列转换为整数类型
df_long['event_count'] = df_long['event_count'].astype(int)

# 查看转换后的数据
print("Long Format DataFrame:")
print(df_long)

# 保存为 Stata 文件（.dta）
df_long.to_csv('program_changed_sport.csv', index=False)
