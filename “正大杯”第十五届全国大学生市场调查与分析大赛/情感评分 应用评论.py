import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

files = {
    r"C:\Users\Lucius\Desktop\NLP\应用 bert\猫箱_bert.xlsx": "#f8d8a4",
    r"C:\Users\Lucius\Desktop\NLP\应用 bert\星野_bert.xlsx": "#ebb089",
    r"C:\Users\Lucius\Desktop\NLP\应用 bert\其它_bert.xlsx": "#588797"
}

bins = np.arange(0, 1.1, 0.1)
bin_centers = bins[:-1] + 0.05
stat_data = {"情感得分区间": [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]}
plt.figure(figsize=(10, 5))

bar_width = 0.03 
for i, (file, color) in enumerate(files.items()):
    df = pd.read_excel(file)
    sentiment_scores = df["sentiment_score"].dropna().values
    hist, _ = np.histogram(sentiment_scores, bins=bins)
    label_name = file.split("\\")[-1].split("_")[0]
    stat_data[label_name] = hist
    plt.bar(bin_centers + i * bar_width, hist, width=bar_width, color=color, edgecolor="black", label=label_name)

stat_df = pd.DataFrame(stat_data)

stat_df.loc["总计"] = ["-"] + [stat_df[col].sum() for col in stat_df.columns[1:]]
print("统计结果：")
print(stat_df)

output_path = r"C:\Users\Lucius\Desktop\应用情感统计.xlsx"
stat_df.to_excel(output_path, index=False)

plt.legend()
plt.xlabel("情感得分区间（负面→正面）")
plt.ylabel("评论数量")
plt.title("应用评论的情感得分分布")
plt.xticks(np.arange(0, 1.1, 0.1))
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()
