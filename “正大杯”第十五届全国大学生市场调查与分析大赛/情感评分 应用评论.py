import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置 Matplotlib 中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 定义文件名和对应的颜色
files = {
    r"C:\Users\Lucius\Desktop\NLP\应用 bert\猫箱_bert.xlsx": "#f8d8a4",
    r"C:\Users\Lucius\Desktop\NLP\应用 bert\星野_bert.xlsx": "#ebb089",
    r"C:\Users\Lucius\Desktop\NLP\应用 bert\其它_bert.xlsx": "#588797"
}

# 设定区间
bins = np.arange(0, 1.1, 0.1)
bin_centers = bins[:-1] + 0.05  # 让不同表的数据并列显示，微调 x 轴位置

# 创建用于存储统计数据的 DataFrame
stat_data = {"情感得分区间": [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]}

# 设置图像
plt.figure(figsize=(10, 5))

# 逐个读取表格数据并绘图
bar_width = 0.03  # 控制每个条形的宽度
for i, (file, color) in enumerate(files.items()):
    # 读取表格
    df = pd.read_excel(file)
    
    # 提取 sentiment_score 列并去除缺失值
    sentiment_scores = df["sentiment_score"].dropna().values
    
    # 统计数据
    hist, _ = np.histogram(sentiment_scores, bins=bins)
    
    # 只保留文件名（去除路径和扩展名）
    label_name = file.split("\\")[-1].split("_")[0]
    
    # 存入统计数据
    stat_data[label_name] = hist
    
    # 绘制条形图
    plt.bar(bin_centers + i * bar_width, hist, width=bar_width, color=color, edgecolor="black", label=label_name)

# 转换成 DataFrame
stat_df = pd.DataFrame(stat_data)

# 计算总计
stat_df.loc["总计"] = ["-"] + [stat_df[col].sum() for col in stat_df.columns[1:]]

# 打印表格
print("统计结果：")
print(stat_df)

# 导出 Excel 文件
output_path = r"C:\Users\Lucius\Desktop\应用情感统计.xlsx"
stat_df.to_excel(output_path, index=False)

# 添加图例
plt.legend()
plt.xlabel("情感得分区间（负面→正面）")
plt.ylabel("评论数量")
plt.title("应用评论的情感得分分布")
plt.xticks(np.arange(0, 1.1, 0.1))
plt.grid(axis="y", linestyle="--", alpha=0.7)

# 显示图表
plt.show()
