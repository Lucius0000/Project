import pandas as pd
import os


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
non_zero_column_threshold = 6

# 打开一个文件来保存文本输出
with open(os.path.join(folder_path_A, 'model_output.txt'), 'w') as output_file:

    # 遍历每个文件并处理
    for file in xlsx_files_A:
        
        if file in xlsx_files_B:  # 匹配同名文件并构建完整路径
            file_path_A = os.path.join(folder_path_A, file)
            file_path_B = os.path.join(folder_path_B, file)


                # 文件夹B中的数据进行预测
            data_B = pd.read_excel(file_path_B)
            X_B = data_B
                
            if X_B.shape[0] == 0:
                print(f"文件 {file} 的 X_B 数据为空，跳过预测过程")

