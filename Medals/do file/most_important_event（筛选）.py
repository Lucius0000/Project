import pandas as pd

# 读取数据
df = pd.read_stata(r"C:\Users\Lucius\Desktop\data_changed\event_participant_medal.dta")

# 1. 计算每个 sport 在 2024 年的出现次数
sport_counts_2024 = df[df['year'] == 2024]['sport'].value_counts()

# 2. 标记出现次数少于 40 次的 sport
sports_to_remove = sport_counts_2024[sport_counts_2024 < 40].index

# 3. 找到在 2024 年根本没出现的 sport
sports_not_in_2024 = df[~df['sport'].isin(sport_counts_2024.index)]['sport'].unique()

# 4. 标记 event_count 在 2024 年小于 10 的 sport
sports_event_count_lt_10 = df[(df['year'] == 2024) & (df['event_count'] < 10)]['sport'].unique()

# 5. 将 sports_event_count_lt_10 合并到 sports_to_remove
sports_to_remove = pd.Index(sports_to_remove).append(pd.Index(sports_not_in_2024)).append(pd.Index(sports_event_count_lt_10))

# 6. 删除这些 sport 在所有年份中的记录
df_filtered = df[~df['sport'].isin(sports_to_remove)]

# 显示或保存结果
# print(df_filtered)  # 可以选择查看结果
df_filtered.to_stata('most_important_event.dta', write_index=False)  # 将结果保存为新的 .dta 文件
