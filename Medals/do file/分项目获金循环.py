# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 23:33:06 2025

@author: Lucius
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
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

pred_file_path = r"C:\Users\Lucius\Desktop\result\pred_2028_gold_sport_forests.xlsx"
performance_file_path = r"C:\Users\Lucius\Desktop\result\performance_2028_gold_sport_forests.xlsx"
importances_path = r"C:\Users\Lucius\Desktop\result\importances_2028_gold_sport_forests"
os.makedirs(importances_path, exist_ok=True)


all_predictions = pd.DataFrame()

all_performance = pd.DataFrame()

# 定义阈值：非全为0的列数大于该值时使用随机森林
non_zero_column_threshold = 0

threshold = 0.05

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
            features = ['participant_count','Country','event_count','host','year_count','gold_previous_1','participants_previous_1','gold_previous_2','participants_previous_2','gold_previous_3','participants_previous_3','gold_previous_4','participants_previous_4','gold_previous_5','participants_previous_5','gold_previous_6','participants_previous_6','gold_previous_7','participants_previous_7','gold_previous_8','participants_previous_8','gold_previous_9','participants_previous_9','gold_previous_10','participants_previous_10','gold_previous_11','participants_previous_11','gold_previous_12','participants_previous_12','gold_previous_13','participants_previous_13','gold_previous_14','participants_previous_14','gold_previous_15','participants_previous_15','gold_previous_16','participants_previous_16','gold_previous_17','participants_previous_17','gold_previous_18','participants_previous_18','gold_previous_19','participants_previous_19','gold_previous_20','participants_previous_20','gold_previous_21','participants_previous_21','gold_previous_22','participants_previous_22','gold_previous_23','participants_previous_23','gold_previous_24','participants_previous_24','gold_previous_25','participants_previous_25','gold_previous_26','participants_previous_26','gold_previous_27','participants_previous_27','gold_previous_28','participants_previous_28','gold_previous_29','participants_previous_29']
            target = 'sport_gold'

            # 数据预处理
            X = data[features]
            y = data[target]
            
            X = X.fillna(0)
            
            # 筛选掉全为 0 的特征列
            X = X.loc[:, (X != 0).any(axis=0)]
            
            # 将 'Country' 列进行独热编码，去掉第一个列（避免虚拟变量陷阱）
            X = pd.get_dummies(X, columns=['Country'], drop_first=True)
            
            new_features = list(X.columns)
            #print(list(new_features))
            
            # 计算非全为 0 的列数
            non_zero_column_count = X.shape[1]
            output_file.write(f"文件: {file}\n")
            output_file.write(f"非全为 0 的列数: {non_zero_column_count}")
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
                #output_file.write(f"文件: {file}\n")
                output_file.write(f"均方误差 (MSE)（优化前）: {mse:.2f}\n")
                output_file.write(f"R^2 值（优化前）: {r2:.2f}\n\n")

                # 进行网格搜索
                grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, scoring='r2', verbose=0, n_jobs=-1)
                grid_search.fit(X_train, y_train)

                # 获取最佳参数
                best_params = grid_search.best_params_
                print(f"最佳参数: {best_params}")
                #output_file.write(f"最佳参数: {best_params}\n")

                # 使用最佳参数重新训练模型
                best_rf_model = grid_search.best_estimator_
                y_pred = best_rf_model.predict(X_test)

                # 计算评价指标
                mse_2 = mean_squared_error(y_test, y_pred)
                r2_2 = r2_score(y_test, y_pred)

                print(f"均方误差 (MSE)（优化后）: {mse_2:.2f}")
                print(f"R^2 值（优化后）: {r2_2:.2f}")
                output_file.write(f"均方误差 (MSE)（优化后）: {mse_2:.2f}\n")
                output_file.write(f"R^2 值（优化后）: {r2_2:.2f}\n\n")
                
                # 判断是否使用随机森林模型
                if r2 < 0.5 and r2_2 < 0.5:
                    print("优化前和优化后的 R² 都小于 0.5，尝试使用线性回归进行预测")
                    output_file.write("优化前和优化后的 R² 都小于 0.5，尝试使用线性回归进行预测\n")
                    
                    reg_model = LinearRegression()
                    reg_model.fit(X_train, y_train)
                    
                    # 使用训练好的模型预测测试集
                    y_pred = reg_model.predict(X_test)

                    # 计算均方误差
                    mse_3 = mean_squared_error(y_test, y_pred)
                    print(f"Mean Squared Error: {mse_3}")
                    output_file.write(f"均方误差 (MSE)（线性回归）: {mse_3:.2f}\n")
                    
                    # 计算 R²
                    r2_3 = r2_score(y_test, y_pred)
                    print(f"R²: {r2}")
                    output_file.write(f"R^2 值（线性回归）: {r2_3:.2f}\n\n")

                    
                    if r2_3 >= max(r2,r2_2):
                        print("线性回归模型R^2更大，使用线性回归进行预测")
                        output_file.write("线性回归模型R^2更大，使用线性回归进行预测\n")
                        r2_use = r2_3
                        mse_use = mse_3
                        # 获取每个特征的系数（特征重要性）
                        feature_importances = reg_model.coef_
                        model_to_use = reg_model
                        
                    else:
                        print("线性回归模型效果更差，使用随机森林进行预测")
                        output_file.write("线性回归模型效果更差，使用随机森林进行预测\n")
                        # 判断是否使用优化后的模型
                        if (r2_2 - r2) > threshold:
                            print("优化后的模型优于优化前的模型，使用优化后的模型进行预测")
                            output_file.write("优化后的模型优于优化前的模型，使用优化后的模型进行预测\n")
                            model_to_use = best_rf_model
                            
                            r2_use = r2_2
                            mse_use = mse_2
                            
                            # 提取特征重要性
                            feature_importances = best_rf_model.feature_importances_
                            sorted_indices = feature_importances.argsort()

                            
                            
                        else:
                            print("优化后的模型未显著优于优化前的模型，使用优化前的模型进行预测")
                            output_file.write("优化后的模型未显著优于优化前的模型，使用优化前的模型进行预测\n")
                            model_to_use = rf_model
                            
                            r2_use = r2
                            mse_use = mse
                            
                            # 提取特征重要性
                            feature_importances = rf_model.feature_importances_
                            sorted_indices = feature_importances.argsort()
                        
                    

                    
                    
                else:
                    # 判断是否使用优化后的模型
                    if (r2_2 - r2) > threshold:
                        print("优化后的模型优于优化前的模型，使用优化后的模型进行预测")
                        model_to_use = best_rf_model
                        
                        r2_use = r2_2
                        mse_use = mse_2
                        
                        # 提取特征重要性
                        feature_importances = best_rf_model.feature_importances_
                        sorted_indices = feature_importances.argsort()

                        
                        
                    else:
                        print("优化后的模型未显著优于优化前的模型，使用优化前的模型进行预测")
                        model_to_use = rf_model
                        
                        r2_use = r2
                        mse_use = mse
                        
                        # 提取特征重要性
                        feature_importances = rf_model.feature_importances_
                        sorted_indices = feature_importances.argsort()
                
                
                
                # 文件夹B中的数据进行预测
                data_B = pd.read_excel(file_path_B)
                
                # 检查 X_B 是否为空
                if data_B.shape[0] == 0:
                    print(f"文件 {file} 的 X_B 数据为空，跳过预测过程")
                    output_file.write(f"文件 {file} 的 X_B 数据为空，跳过预测过程\n")
                    continue  # 跳过当前文件，进入下一个文件
                
                X_B = pd.get_dummies(data_B, columns=['Country'], drop_first=True)
                
                X_B = X_B[[col for col in new_features if col in X_B.columns]]
                
                # 获取训练时使用的特征列
                train_columns = X.columns
                # 获取预测数据中的列
                pred_columns = X_B.columns
                # 找到缺失的特征列，并补齐为 0
                missing_columns = set(train_columns) - set(pred_columns)
                for col in missing_columns:
                    X_B[col] = 0
                # 确保预测数据的列顺序和训练时一致
                X_B = X_B[train_columns]

                             
                y_pred_B = model_to_use.predict(X_B)
                
                # 将预测结果添加到DataFrame中
                predictions_df = pd.DataFrame({
                    'Country':data_B['Country'],
                    'noc':data_B['noc'],
                    'Year':2028,
                    'sport':data_B['sport'],
                    'Pred_Gold': y_pred_B,
                })
                
                
                all_predictions = pd.concat([all_predictions, predictions_df], ignore_index=True)
                
                
                performance = []
                
                # 获取特征重要性并保存
                sorted_indices = feature_importances.argsort()
                
                for idx in sorted_indices[::-1]:  # 从高到低排序输出
                    feature_name = X.columns[idx]
                    importance = feature_importances[idx]
                    
                    sport = data_B['sport'].iloc[0]
                    
                    # 保存每个特征的 R² 和 MSE
                    performance.append({
                        "sport": data_B['sport'].iloc[0],
                        "r2": r2_use,
                        "mse": mse_use,
                        "feature": feature_name,
                        "feature_importance": importance
                    })
                    
                performance_df = pd.DataFrame(performance, columns=['sport', 'r2', 'mse', 'feature', 'feature_importance'])
                    
                # 过滤掉特征重要性为0的特征
                non_zero_importances = performance_df[performance_df['feature_importance'] != 0]
                
                all_performance = pd.concat([all_performance,non_zero_importances],ignore_index=True) 
                
                
                '''
                # 绘制特征重要性条形图
                if len(non_zero_importances) > 0:  # 如果有特征重要性
                    sorted_features = non_zero_importances['feature']
                    sorted_importances = non_zero_importances['feature_importance']
                    
                    # 自适应调整行列数
                    num_features = len(sorted_features)
                    ncols = 3  # 设置每行展示3个特征
                    nrows = (num_features // ncols) + (1 if num_features % ncols != 0 else 0)  # 根据特征数决定行数

                    # 设置支持中文的字体
                    rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
                    rcParams['axes.unicode_minus'] = False  # 防止负号显示问题

                    # 绘制条形图
                    plt.figure(figsize=(ncols * 5, nrows * 3))  # 图像大小根据特征数量调整
                    plt.barh(sorted_features, sorted_importances, align='center')
                    plt.xlabel("特征重要性")
                    plt.title(f"{sport} 特征重要性")
                    
                    # 保存图像
                    plt.tight_layout()  # 自动调整布局
                    img_output_path = os.path.join(importances_path, f"feature_importance_{sport}.png")
                    plt.savefig(img_output_path)
                    plt.close()

                    print(f"Feature importance plot for {sport} saved as {img_output_path}")
                    '''
                
                print(f"文件 {file} 的预测结果已添加。")
                

            else:
                print(f"非全为 0 的列数未超过阈值 {non_zero_column_threshold}，转用其他模型")
                output_file.write(f"非全为 0 的列数未超过阈值 {non_zero_column_threshold}，转用其他模型")
                
        else:
            print(f"文件夹A中的文件 {file} 在文件夹B中找不到对应文件，跳过预测。")
            
# 保存所有预测结果到一个Excel文件
with pd.ExcelWriter(pred_file_path, mode='w', engine='openpyxl') as writer:
    all_predictions.to_excel(writer, index=False, sheet_name='Predictions')
    
with pd.ExcelWriter(performance_file_path, mode='w', engine='openpyxl') as writer:
    all_performance.to_excel(writer, index=False, sheet_name='Performance')

print(f"所有预测结果已保存到 {pred_file_path}")
                


            

        
        

        
        
        
        
        
        
        
        