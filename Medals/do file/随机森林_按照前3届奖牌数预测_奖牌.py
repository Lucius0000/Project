# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 18:08:39 2025

@author: Lucius
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 17:44:43 2025

@author: Lucius
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 加载数据
data = pd.read_excel(r"C:\Users\Lucius\Desktop\变量(2).xlsx")  # 替换为实际数据路径

# 示例特征：GDP、人口、历史奖牌数、参赛项目数等
features = ['Historical_Gold','Historical_Total','num_participants','num_sports','num_events','Host']
target = 'Total'

# 数据预处理
X = data[features]
y = data[target]

# 将分类特征编码（如果有需要）
#X = pd.get_dummies(X, columns=['Host_Country'], drop_first=True)

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化随机森林模型
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# 训练模型
rf_model.fit(X_train, y_train)

# 模型预测
y_pred = rf_model.predict(X_test)

# 计算评价指标
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"均方误差 (MSE): {mse:.2f}")
print(f"R^2 值: {r2:.2f}")

import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置支持中文的字体
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
rcParams['axes.unicode_minus'] = False  # 防止负号显示问题

# 提取特征重要性
feature_importances = rf_model.feature_importances_
sorted_indices = feature_importances.argsort()

# 绘制特征重要性条形图
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_indices)), feature_importances[sorted_indices], align='center')
plt.yticks(range(len(sorted_indices)), [X.columns[i] for i in sorted_indices])
plt.xlabel("特征重要性")
plt.title("随机森林特征重要性")
plt.show()


from sklearn.model_selection import GridSearchCV

# 定义超参数范围
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 网格搜索
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
                           param_grid=param_grid,
                           cv=3,
                           scoring='neg_mean_squared_error',
                           verbose=2,
                           n_jobs=-1)

grid_search.fit(X_train, y_train)

# 输出最佳参数
print("最佳参数:", grid_search.best_params_)

# 使用最佳参数重新训练模型
best_rf_model = grid_search.best_estimator_


# 用最佳模型在测试集上进行预测
y_pred_best = best_rf_model.predict(X_test)

# 计算新的评价指标
mse_best = mean_squared_error(y_test, y_pred_best)
r2_best = r2_score(y_test, y_pred_best)

print(f"使用最佳参数后的均方误差 (MSE): {mse_best:.2f}")
print(f"使用最佳参数后的R^2 值: {r2_best:.2f}")


#需要变换2028-8的列名，并加入host列
full_data2028 = pd.read_excel(r"C:\Users\Lucius\Desktop\do file\2028_8.xlsx")
data2028 = full_data2028[features]

y_pred_2028 = best_rf_model.predict(data2028)
y_pred_2028 = np.round(y_pred_2028).astype(int) 

pred_2028 = pd.DataFrame({
    'country':full_data2028['Country'],
    'year':2028,
    'pred_medal':y_pred_2028
    })


pred_2028.to_excel('pred_2028_依据8届平均_medal.xlsx',index=False)
print('success to save pred_file')



