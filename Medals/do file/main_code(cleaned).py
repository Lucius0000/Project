import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import os
from sklearn.model_selection import GridSearchCV
folder_path_A = r"C:\Users\Lucius\Desktop\data\train"
folder_path_B = r"C:\Users\Lucius\Desktop\data\pred"  
xlsx_files_A = [f for f in os.listdir(folder_path_A) if f.endswith('.xlsx')]
xlsx_files_B = [f for f in os.listdir(folder_path_B) if f.endswith('.xlsx')]
pred_file_path = r"C:\Users\Lucius\Desktop\result\pred_2028_gold_sport_forests.xlsx"
performance_file_path = r"C:\Users\Lucius\Desktop\result\performance_2028_gold_sport_forests.xlsx"
all_predictions = pd.DataFrame()
all_performance = pd.DataFrame()
non_zero_column_threshold = 1
threshold = 0.05
with open(os.path.join(folder_path_A, 'model_output.txt'), 'w') as output_file:
    for file in xlsx_files_A:
        if file in xlsx_files_B: 
            file_path_A = os.path.join(folder_path_A, file)
            file_path_B = os.path.join(folder_path_B, file)
            data = pd.read_excel(file_path_A) 
            if data.shape[0] < 10:
                print("The sample size is less than 10, skip using the Random Forest model.")
                output_file.write("The sample size is less than 10, skip using the Random Forest model.")
                continue
            features = ['participant_count','Country','event_count','host','year_count','gold_previous_1','participants_previous_1','gold_previous_2','participants_previous_2','gold_previous_3','participants_previous_3','gold_previous_4','participants_previous_4','gold_previous_5','participants_previous_5','gold_previous_6','participants_previous_6','gold_previous_7','participants_previous_7','gold_previous_8','participants_previous_8','gold_previous_9','participants_previous_9','gold_previous_10','participants_previous_10','gold_previous_11','participants_previous_11','gold_previous_12','participants_previous_12','gold_previous_13','participants_previous_13','gold_previous_14','participants_previous_14','gold_previous_15','participants_previous_15','gold_previous_16','participants_previous_16','gold_previous_17','participants_previous_17','gold_previous_18','participants_previous_18','gold_previous_19','participants_previous_19','gold_previous_20','participants_previous_20','gold_previous_21','participants_previous_21','gold_previous_22','participants_previous_22','gold_previous_23','participants_previous_23','gold_previous_24','participants_previous_24','gold_previous_25','participants_previous_25','gold_previous_26','participants_previous_26','gold_previous_27','participants_previous_27','gold_previous_28','participants_previous_28','gold_previous_29','participants_previous_29']
            target = 'sport_gold'
            X = data[features]
            y = data[target]
            X = X.fillna(0)
            X = X.loc[:, (X != 0).any(axis=0)]
            X = pd.get_dummies(X, columns=['Country'], drop_first=True)
            new_features = list(X.columns)
            non_zero_column_count = X.shape[1]
            if non_zero_column_count > non_zero_column_threshold:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                param_grid = {
                    'n_estimators': [50, 100, 200],  
                    'max_depth': [None, 10, 20, 30], 
                    'min_samples_split': [2, 5, 10], 
                    'min_samples_leaf': [1, 2, 4],    
                }
                rf_model = RandomForestRegressor(random_state=42)
                rf_model.fit(X_train, y_train)
                y_pred = rf_model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, scoring='r2', verbose=0, n_jobs=-1)
                grid_search.fit(X_train, y_train)
                best_params = grid_search.best_params_
                best_rf_model = grid_search.best_estimator_
                y_pred = best_rf_model.predict(X_test)
                mse_2 = mean_squared_error(y_test, y_pred)
                r2_2 = r2_score(y_test, y_pred)
                if r2 < 0.5 and r2_2 < 0.5:
                    reg_model = LinearRegression()
                    reg_model.fit(X_train, y_train)
                    y_pred = reg_model.predict(X_test)
                    mse_3 = mean_squared_error(y_test, y_pred)
                    r2_3 = r2_score(y_test, y_pred)
                    if r2_3 >= max(r2,r2_2):
                        r2_use = r2_3
                        mse_use = mse_3
                        feature_importances = reg_model.coef_
                        model_to_use = reg_model      
                    else:
                        if (r2_2 - r2) > threshold:
                            model_to_use = best_rf_model 
                            r2_use = r2_2
                            mse_use = mse_2
                            feature_importances = best_rf_model.feature_importances_
                            sorted_indices = feature_importances.argsort()                  
                        else:
                            model_to_use = rf_model
                            r2_use = r2
                            mse_use = mse
                            feature_importances = rf_model.feature_importances_
                            sorted_indices = feature_importances.argsort()           
                else:
                    if (r2_2 - r2) > threshold:
                        model_to_use = best_rf_model
                        r2_use = r2_2
                        mse_use = mse_2
                        feature_importances = best_rf_model.feature_importances_
                        sorted_indices = feature_importances.argsort()
                    else:
                        model_to_use = rf_model
                        r2_use = r2
                        mse_use = mse
                        feature_importances = rf_model.feature_importances_
                        sorted_indices = feature_importances.argsort()
                data_B = pd.read_excel(file_path_B)
                if data_B.shape[0] == 0:
                    continue
                X_B = pd.get_dummies(data_B, columns=['Country'], drop_first=True)
                X_B = X_B[[col for col in new_features if col in X_B.columns]]
                train_columns = X.columns
                pred_columns = X_B.columns
                missing_columns = set(train_columns) - set(pred_columns)
                for col in missing_columns:
                    X_B[col] = 0
                X_B = X_B[train_columns]
                y_pred_B = model_to_use.predict(X_B)
                predictions_df = pd.DataFrame({
                    'Country':data_B['Country'],
                    'noc':data_B['noc'],
                    'Year':2028,
                    'sport':data_B['sport'],
                    'Pred_Gold': y_pred_B,
                })
                all_predictions = pd.concat([all_predictions, predictions_df], ignore_index=True)
                performance = []
                sorted_indices = feature_importances.argsort()
                for idx in sorted_indices[::-1]: 
                    feature_name = X.columns[idx]
                    importance = feature_importances[idx]
                    sport = data_B['sport'].iloc[0]
                    performance.append({
                        "sport": data_B['sport'].iloc[0],
                        "r2": r2_use,
                        "mse": mse_use,
                        "feature": feature_name,
                        "feature_importance": importance
                    })    
                performance_df = pd.DataFrame(performance, columns=['sport', 'r2', 'mse', 'feature', 'feature_importance'])
                non_zero_importances = performance_df[performance_df['feature_importance'] != 0]
                all_performance = pd.concat([all_performance,non_zero_importances],ignore_index=True)    
        else:
            print('file not found in B, skip')
with pd.ExcelWriter(pred_file_path, mode='w', engine='openpyxl') as writer:
    all_predictions.to_excel(writer, index=False, sheet_name='Predictions')   
with pd.ExcelWriter(performance_file_path, mode='w', engine='openpyxl') as writer:
    all_performance.to_excel(writer, index=False, sheet_name='Performance')
print('Done.')