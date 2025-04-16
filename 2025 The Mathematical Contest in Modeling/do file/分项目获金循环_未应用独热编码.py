# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 01:59:37 2025

@author: Lucius
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.model_selection import GridSearchCV


# 设置文件夹路径
folder_path_A = r"C:\Users\Lucius\Desktop\数据(1)\分项目获金数据合集"
folder_path_B = r"C:\Users\Lucius\Desktop\数据(1)\2028预测数据获金"  # 替换为你的文件夹路径

# 获取文件夹中所有.xlsx文件
xlsx_files_A = [f for f in os.listdir(folder_path_A) if f.endswith('.xlsx')]
#print (xlsx_files_A)

xlsx_files_B = [f for f in os.listdir(folder_path_B) if f.endswith('.xlsx')]

pred_file_path = r"C:\Users\Lucius\Desktop\result\pred_2028_sport_forests.xlsx"

all_predictions = pd.DataFrame()

# 定义阈值：非全为0的列数大于该值时使用随机森林
non_zero_column_threshold = 0

# 打开一个文件来保存文本输出
with open(os.path.join(folder_path_A, 'model_output.txt'), 'w') as output_file:

    # 遍历每个文件并处理
    for file in xlsx_files_A:
        
        if file in xlsx_files_B:  # 匹配同名文件并构建完整路径
            file_path_A = os.path.join(folder_path_A, file)
            file_path_B = os.path.join(folder_path_B, file)
            
            # 加载数据
            data = pd.read_excel(file_path_A)  # 替换为实际数据路径
            
            if data.shape[0] < 10:
                print(f"文件 {file} 的样本数小于10，跳过使用随机森林模型")
                output_file.write(f"文件 {file} 的样本数小于10，跳过使用随机森林模型\n")
                continue  # 跳过当前文件，进入下一个文件

            #以下是随机森林代码

            # 示例特征：GDP、人口、历史奖牌数、参赛项目数等
            features = ['participant_count','event_count','host','year_count','gold_previous_1','participants_previous_1','gold_previous_2','participants_previous_2','gold_previous_3','participants_previous_3','gold_previous_4','participants_previous_4','gold_previous_5','participants_previous_5','gold_previous_6','participants_previous_6','gold_previous_7','participants_previous_7','gold_previous_8','participants_previous_8','gold_previous_9','participants_previous_9','gold_previous_10','participants_previous_10','gold_previous_11','participants_previous_11','gold_previous_12','participants_previous_12','gold_previous_13','participants_previous_13','gold_previous_14','participants_previous_14','gold_previous_15','participants_previous_15','gold_previous_16','participants_previous_16','gold_previous_17','participants_previous_17','gold_previous_18','participants_previous_18','gold_previous_19','participants_previous_19','gold_previous_20','participants_previous_20','gold_previous_21','participants_previous_21','gold_previous_22','participants_previous_22','gold_previous_23','participants_previous_23','gold_previous_24','participants_previous_24','gold_previous_25','participants_previous_25','gold_previous_26','participants_previous_26','gold_previous_27','participants_previous_27','gold_previous_28','participants_previous_28','gold_previous_29','participants_previous_29']
            target = 'sport_gold'

            # 数据预处理
            X = data[features]
            y = data[target]
            
            X = X.fillna(0)
            
            # 筛选掉全为 0 的特征列
            X = X.loc[:, (X != 0).any(axis=0)]
            
            new_features = list(X.columns)
            #print(set(new_features))
            
            # 计算非全为 0 的列数
            non_zero_column_count = X.shape[1]
            output_file.write(f"文件: {file}, 非全为 0 的列数: {non_zero_column_count}")
            print(f"文件: {file}, 非全为 0 的列数: {non_zero_column_count}")

            
            # 检查被移除的特征列
            '''
            removed_features = set(features) - set(X.columns)
            if removed_features:
                print(f"被移除的特征: {removed_features}")
                output_file.write(f"文件: {file}\n")
                output_file.write(f"被移除的特征: {removed_features}\n")
                '''
    
            # 根据非全为 0 的列数选择模型
            if non_zero_column_count > non_zero_column_threshold:
                print(f"非全为 0 的列数超过阈值 {non_zero_column_threshold}，使用随机森林")
                output_file.write(f"非全为 0 的列数超过阈值 {non_zero_column_threshold}，使用随机森林")
                
                # 数据集划分
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # 定义网格搜索参数
                param_grid = {
                    'n_estimators': [50, 100, 200],      # 决策树个数
                    'max_depth': [None, 10, 20, 30],    # 最大深度
                    'min_samples_split': [2, 5, 10],    # 最小分裂样本数
                    'min_samples_leaf': [1, 2, 4],      # 叶子节点的最小样本数
                }

                # 初始化随机森林模型
                rf_model = RandomForestRegressor(random_state=42)
                
                # 训练模型
                rf_model.fit(X_train, y_train)

                # 模型预测
                y_pred = rf_model.predict(X_test)

                # 计算评价指标
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                print(f"均方误差 (MSE)（优化前）: {mse:.2f}")
                print(f"R^2 值（优化前）: {r2:.2f}")
                
                # 输出文本到文件
                output_file.write(f"文件: {file}\n")
                output_file.write(f"均方误差 (MSE)（优化前）: {mse:.2f}\n")
                output_file.write(f"R^2 值（优化前）: {r2:.2f}\n\n")

                # 进行网格搜索
                grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, scoring='r2', verbose=0, n_jobs=-1)
                grid_search.fit(X_train, y_train)

                # 获取最佳参数
                best_params = grid_search.best_params_
                print(f"最佳参数: {best_params}")
                output_file.write(f"最佳参数: {best_params}\n")

                # 使用最佳参数重新训练模型
                best_rf_model = grid_search.best_estimator_
                y_pred = best_rf_model.predict(X_test)

                # 计算评价指标
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                print(f"均方误差 (MSE)（优化后）: {mse:.2f}")
                print(f"R^2 值（优化后）: {r2:.2f}")
                output_file.write(f"均方误差 (MSE)（优化后）: {mse:.2f}\n")
                output_file.write(f"R^2 值（优化后）: {r2:.2f}\n\n")

                # 提取特征重要性
                feature_importances = best_rf_model.feature_importances_
                sorted_indices = feature_importances.argsort()

                # 输出特征重要性到文本文件
                output_file.write("特征重要性排名 (最佳参数模型):\n")
                for idx in sorted_indices[::-1]:  # 从高到低排序输出
                    feature_name = X.columns[idx]
                    importance = feature_importances[idx]
                    output_file.write(f"{feature_name}: {importance:.4f}\n")
                output_file.write("\n")
                
                
                # 设置支持中文的字体
                rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
                rcParams['axes.unicode_minus'] = False  # 防止负号显示问题

                # 绘制特征重要性条形图
                plt.figure(figsize=(10, 10))
                plt.barh(range(len(sorted_indices)), feature_importances[sorted_indices], align='center')
                plt.yticks(range(len(sorted_indices)), [X.columns[i] for i in sorted_indices])
                plt.xlabel("特征重要性")
                plt.title("随机森林特征重要性（最佳参数模型）")

                # 保存图像
                img_output_path = os.path.join(folder_path_A, f"{file}_feature_importance_optimized.png")
                plt.savefig(img_output_path)
                plt.close()            # 关闭当前图形，销毁图形对象，释放内存

                print(f"Feature importance plot (optimized) saved as {img_output_path}")
                
                # 文件夹B中的数据进行预测
                data_B = pd.read_excel(file_path_B)
                
                # 检查 X_B 是否为空
                if data_B.shape[0] == 0:
                    print(f"文件 {file} 的 X_B 数据为空，跳过预测过程")
                    output_file.write(f"文件 {file} 的 X_B 数据为空，跳过预测过程\n")
                    continue  # 跳过当前文件，进入下一个文件

                X_B = data_B[new_features]

                y_pred_B = best_rf_model.predict(X_B)
                
                # 将预测结果添加到DataFrame中
                predictions_df = pd.DataFrame({
                    'Country':data_B['Country'],
                    'noc':data_B['noc'],
                    'Year':2028,
                    'sport':data_B['sport'],
                    'Pred_Gold': y_pred_B,  # 预测值列
                })

                all_predictions = pd.concat([all_predictions, predictions_df], ignore_index=True)
                
                print(f"文件 {file} 的预测结果已添加。")
                

            else:
                print(f"非全为 0 的列数未超过阈值 {non_zero_column_threshold}，转用其他模型")
                output_file.write(f"非全为 0 的列数未超过阈值 {non_zero_column_threshold}，转用其他模型")
                
        else:
            print(f"文件夹A中的文件 {file} 在文件夹B中找不到对应文件，跳过预测。")
            
# 保存所有预测结果到一个Excel文件
with pd.ExcelWriter(pred_file_path, mode='w', engine='openpyxl') as writer:
    all_predictions.to_excel(writer, index=False, sheet_name='Predictions')

print(f"所有预测结果已保存到 {pred_file_path}")
                


            

        
        

        
        
        
        
        
        
        
        