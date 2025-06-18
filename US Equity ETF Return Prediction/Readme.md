## US Equity ETF Return Prediction

---

以下是文件说明：

- 完整代码：`ETF收益预测模型.ipynb`

- 研究报告：`基于多模型横截面IC的ETF收益预测研究：传统计量、机器学习与深度学习的比较分析.pdf`
- 代码依赖库：`requirements.txt`
- `IC图表.xlsx`和`IC统计值.xlsx`是模型结果，报告中有贴上去、也有详细解释，但是受篇幅所限可能不够清晰，所有在此附上
- 数据来源是Akshare，`etf_data_raw`文件夹存放了原始数据，合并、清洗后的数据在`etf_data_processed`文件夹下的`etf_daily_data_2015_2024_clean.csv`，构建特征工程后的输入、回测数据是`etf_data_processed`文件夹下的`X_static_window7_scaled.csv`,`Y_static_window7.csv`,`X_sequence_window7_scaled.npy`,`Y_sequence_window7.npy`，有四个文件是因为构建了两种滑动窗口，把特征值和标签值分开存放了，所以有2*2=4；训练与回测数据没有显式分开存放，在每一次建模都以时间顺序、8比2划分了，2023 年 9 月 28 日前（含）是训练数据，其后是回测数据。
- `etf_data_processed`文件夹下的带`_result`后缀的子文件夹存放了各个模型的结果评估
