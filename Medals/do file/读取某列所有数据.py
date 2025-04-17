import pandas as pd

# 读取CSV文件
df = pd.read_csv(r"C:\Users\Lucius\Desktop\零的突破.csv")

# 选择你想要的列，例如 'Country'（注意拼写）
selected_columns = ['country']  # 修改列名为正确的列名
df_selected = df[selected_columns]

# 遍历每一行并格式化为 '信息A', '信息B'（用逗号连接，不换行）
formatted_data = []
for index, row in df_selected.iterrows():
    formatted_row = ",".join([f"'{str(value)}'" for value in row])  # 用逗号连接且用单引号包围
    formatted_data.append(formatted_row)

# 合并所有行并用逗号分隔（不换行）
final_output = ", ".join(formatted_data)

# 打印结果
print(final_output)

'''
# 将格式化的数据保存到文件
with open('formatted_data.txt', 'w') as f:
    f.write(final_output)
'''