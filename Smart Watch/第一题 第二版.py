import os
import pandas as pd
import re
import warnings
import gc

# 禁用所有警告
warnings.filterwarnings('ignore')

# 设置路径
input_folder = r'C:\Users\Lucius\Downloads\2025泰迪杯 赛题\B\B题-全部数据\附件1'
output_folder = r'C:\Users\Lucius\Downloads\2025泰迪杯 赛题\B\结果\附件1'
summary_file = r'C:\Users\Lucius\Downloads\2025泰迪杯 赛题\B\结果\附件1\label_summary.csv'
empty_value_summary_file = r'C:\Users\Lucius\Downloads\2025泰迪杯 赛题\B\结果\附件1\empty_value_summary.csv'
met_summary_file = r'C:\Users\Lucius\Downloads\2025泰迪杯 赛题\B\结果\附件1\met_summary.csv'

os.makedirs(output_folder, exist_ok=True)

summary_dict = {}
empty_value_summary_list = []
met_summary_list = []

# 根据MET值区间进行分类
def categorize_met(met_value):
    if met_value >= 6.0:
        return 'High Intensity'
    elif 3.0 <= met_value < 6.0:
        return 'Moderate Intensity'
    elif 1.6 <= met_value < 3.0:
        return 'Low Intensity'
    elif 1.0 <= met_value < 1.6:
        return 'Static Behavior'
    elif met_value < 1:
        return 'Sleep'
    else:
        return 'Unknown'

# 遍历以P开头的csv文件
for file_name in os.listdir(input_folder):
    if file_name.startswith('P') and file_name.endswith('.csv'):
        file_path = os.path.join(input_folder, file_name)

        df = pd.read_csv(file_path, usecols=['annotation'])

        # 统计annotation列为空的比例
        empty_annotation_count = df['annotation'].isna().sum()
        total_rows = len(df)
        empty_annotation_ratio = empty_annotation_count / total_rows * 100

        # 存储空值统计信息
        empty_value_summary_list.append({
            'file': file_name,
            'empty_annotation_count': empty_annotation_count,
            'total_rows': total_rows,
            'empty_annotation_ratio (%)': empty_annotation_ratio
        })

        # 处理非空的annotation列
        df_non_empty = df.dropna(subset=['annotation'])

        # 拆分annotation列，确保拆分成功并创建新的列
        df_non_empty[['label', 'MET']] = df_non_empty['annotation'].str.rsplit(';', n=1, expand=True)

        # 检查拆分是否成功
        if 'label' not in df_non_empty.columns:
            print(f"Warning: 'label' column not found in {file_name}. Skipping file.")
            continue  # 如果拆分失败，跳过当前文件

        # 提取 MET 列中的数字部分
        def extract_met_value(met_str):
            match = re.search(r'MET\s*([\d\.]+)', met_str)
            if match:
                return float(match.group(1))
            else:
                return None

        df_non_empty['MET'] = df_non_empty['MET'].apply(extract_met_value)

        # 删除原annotation列
        df_non_empty.drop(columns=['annotation'], inplace=True)

        # 统计label频次
        label_counts = df_non_empty['label'].value_counts().reset_index()
        label_counts.columns = ['label', file_name]  # 用文件名做列名
        summary_dict[file_name] = label_counts.set_index('label')[file_name]

        # MET值按区间统计
        df_non_empty['MET_Category'] = df_non_empty['MET'].apply(categorize_met)

        # 计算每个区间的频次
        met_counts = df_non_empty['MET_Category'].value_counts().reset_index()
        met_counts.columns = ['MET_Category', f'{file_name}_Count']

        # 计算每个区间的总时长（单位：小时）
        met_duration = df_non_empty.groupby('MET_Category').size() / 100 / 3600  # 总观测数 / 每秒100次采样 / 3600秒(小时)
        met_duration = met_duration.round(4).reset_index()
        met_duration.columns = ['MET_Category', f'{file_name}_Duration_Hours']

        # 合并频次和时长
        met_stats = pd.merge(met_counts, met_duration, on='MET_Category', how='outer')
        
        #添加一个单独的总时长记录（Total_Duration_Hours），也按格式写入
        total_duration_hours = df_non_empty.shape[0] / 100 / 3600  # 每秒100次采样
        total_duration_row = pd.DataFrame({
            'MET_Category': ['Total_Duration_Hours'],
            f'{file_name}_Count': [df_non_empty.shape[0]],
            f'{file_name}_Duration_Hours': [round(total_duration_hours, 4)]
        })

        #  合并 total_duration_row 到 met_stats
        met_stats = pd.concat([met_stats, total_duration_row], ignore_index=True)
        
        met_stats['file'] = file_name
        met_summary_list.append(met_stats)

        # 计算文件的总时长
        total_duration_hours = df_non_empty.shape[0] / 100 / 3600  # 每秒100次采样
        met_stats['total_duration_hours'] = total_duration_hours

        print(f"已处理: {file_name}")
        
        del df
        gc.collect()

# 用于合并所有文件的 MET 分类统计
met_summary_df = None

for met_stats in met_summary_list:
    file_name = met_stats['file'].iloc[0]  # 当前处理文件名

    # 以 MET_Category 为索引，准备横向合并
    met_stats = met_stats.set_index('MET_Category')[[f'{file_name}_Count', f'{file_name}_Duration_Hours']]
    met_stats.columns = [f'{file_name}_Count', f'{file_name}_Duration_Hours']

    if met_summary_df is None:
        met_summary_df = met_stats
    else:
        met_summary_df = pd.merge(met_summary_df, met_stats, how='outer', left_index=True, right_index=True)

# 重置索引，使 MET_Category 成为列
met_summary_df = met_summary_df.reset_index()

# 填充缺失值并保留小数
met_summary_df = met_summary_df.fillna(0).round(4)

# 保存结果
met_summary_df.to_csv(met_summary_file, index=False)
print(f"\nMET区间统计已输出: {met_summary_file}")


# 合并所有文件的label统计结果
final_summary_df = pd.DataFrame(summary_dict).fillna(0).astype(int)
final_summary_df.to_csv(summary_file)
print(f"\n标签频次统计已输出: {summary_file}")

# 将空值统计结果保存为CSV
empty_value_summary_df = pd.DataFrame(empty_value_summary_list)
empty_value_summary_df.to_csv(empty_value_summary_file, index=False)
print(f"\n空值统计结果已输出: {empty_value_summary_file}")
