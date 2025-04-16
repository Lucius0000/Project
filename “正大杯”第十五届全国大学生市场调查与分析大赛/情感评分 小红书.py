# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 14:46:46 2025

@author: Lucius
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置 Matplotlib 中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

df = pd.read_excel(r"C:\Users\Lucius\Desktop\NLP\小红书_bert.xlsx")

# 设定区间
bins = np.arange(0, 1.1, 0.1)
bin_centers = bins[:-1] + 0.05  # 让不同表的数据并列显示，微调 x 轴位置

# 处理标题、正文和评论的情感评分
title_scores = df['标题情感评分'].dropna().unique()
body_scores = df['正文情感评分'].dropna().unique()
comment_scores = df['评论情感评分'].dropna().unique()

# 统计数据
title_hist, _ = np.histogram(title_scores, bins=bins)
body_hist, _ = np.histogram(body_scores, bins=bins)
comment_hist, _ = np.histogram(comment_scores, bins=bins)

# 统计总数
total_titles = len(title_scores)
total_bodies = len(body_scores)
total_comments = len(comment_scores)

# 创建统计结果的表格
stat_df = pd.DataFrame({
    "情感得分区间": [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)],
    "标题数量": title_hist,
    "正文数量": body_hist,
    "评论数量": comment_hist
})

# 添加总数统计
stat_df.loc["总计"] = ["-", total_titles, total_bodies, total_comments]

# 打印表格
print("统计结果：")
print(stat_df)

# 也可以导出 Excel 或 CSV 以便查看
stat_df.to_excel(r"C:\Users\Lucius\Desktop\情感统计.xlsx", index=False)
# stat_df.to_csv(r"C:\Users\Lucius\Desktop\情感统计.csv", index=False, encoding="utf-8-sig")

# 设置图像
fig, ax1 = plt.subplots(figsize=(10, 5))

# 绘制标题和正文的条形图，使用左 y 轴
bar_width = 0.03  # 控制每个条形的宽度
ax1.bar(bin_centers, title_hist, width=bar_width, color="#f8d8a4", edgecolor="black", label="标题", alpha=0.8)
ax1.bar(bin_centers + bar_width, body_hist, width=bar_width, color="#ebb089", edgecolor="black", label="正文", alpha=0.8)

# 设置左 y 轴的标签
ax1.set_xlabel("情感得分区间（负面→正面）")
ax1.set_ylabel("标题/正文数量")
ax1.set_xticks(np.arange(0, 1.1, 0.1))
ax1.grid(axis="y", linestyle="--", alpha=0.7)

# 添加图例到左 y 轴
ax1.legend(loc="upper left")

# 创建右 y 轴，使用它来绘制评论的条形图
ax2 = ax1.twinx()
ax2.bar(bin_centers + 2 * bar_width, comment_hist, width=bar_width, color="#588797", edgecolor="black", label="评论", alpha=0.8)
ax2.set_ylabel("评论数量")  # 右 y 轴的标签

# **调整右 y 轴范围，避免评论的柱形图与其他重叠**
ax2.set_ylim(0, max(comment_hist) * 1.2)  

# **调整图例位置，防止重叠**
ax2.legend(loc="upper right", bbox_to_anchor=(1, 0.9))

# 设置标题
plt.title("小红书帖子情感得分分布")

# 显示图表
plt.show()
