# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 22:43:23 2025

@author: Lucius
"""

import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import tkinter as tk
from tkinter import filedialog
import sys
from factor_analyzer import FactorAnalyzer
from semopy import Model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
from scipy import stats
from scipy.stats import norm
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from semopy import calc_stats
from graphviz import Digraph
import warnings

warnings.filterwarnings('ignore')
    
    
import stata_setup
stata_setup.config(r"C:\Program Files\Stata18","mp") #只填写路径而非exe

from pystata import stata


root = tk.Tk()
root.withdraw()


print("请选择用户数据文件（仅支持Excel）...")
file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx;*.xls")])
if not file_path:
    print("未选择文件，程序退出。")
    sys.exit()
    
print("请选择输出文件夹...")
output_path = filedialog.askdirectory()
if not output_path:
    print("未选择输出文件夹，程序退出。")
    sys.exit()

r'''
file_path = r"C:\Users\Lucius\Desktop\市调赛;Z世代对情感陪伴AI的消费态度与市场潜力研究\问卷数据处理\问卷数据 原始整合.xlsx"
output_path = r"C:\Users\Lucius\Desktop\output"
'''

os.makedirs(output_path, exist_ok = True)
try:
    df = pd.read_excel(file_path)
except FileNotFoundError:
    print("错误：未找到指定的文件，请检查文件路径。")
    sys.exit()
else:
    ''' 数据清洗 '''
    
    #删去列
    columns_to_drop = ['序号', '提交答卷时间', '所用时间', '来源', '来源详情', '来自IP', '总分']
    df = df.drop(columns=columns_to_drop, errors='ignore')
    #print(df.columns)
    #重命名列名
    #数据字典，留意有的列名存在不合理的空格
    rename_dict = {
        '您的性别?': 'Gender',
        '您的年龄?': 'Age',
        '您目前的身份?': 'Identity',
        '您的月消费额是?': 'Spending',
        '您是否会感到孤独？': 'Loneliness',
        '您是否会感到情绪诅丧': 'Depre',
        '当您需要情感支持时，您主要依靠什么方式？ ': 'Support',
        '您是否使用过情感陪伴AI?💡 提示： 情感陪伴AI指的是能够与用户进行互动交流，并提供情感支持或陪伴的人工智能产品。例如：   聊天机器人（如小爱同学、豆包智能体、QQ小冰）  心理陪伴AI（如星野、猫箱） 智能陪伴应用（如 Replika、男友/女友AI）  具备情感互动功能的虚拟角色（如部分游戏中的AI角色、虚拟偶像）': 'UsedFreq',
        '您使用情感陪伴AI的主要原因?(缓解孤独)': 'ReasonLonely',
        '8(寻求情感寄托)': 'ReasonEmo',
        '8(获取心理支持)': 'ReasonSupport',
        '8(练习社交、沟通能力)': 'ReasonSocial',
        '8(消遣娱乐)': 'ReasonEnter',
        '8(新奇体验)': 'ReasonNovel',
        '8(其他（请说明）)': 'ReasonOther',
        '您对情感陪伴AI的使用体验如何？—整体体验': 'ExperOverall',
        '对话自然度': 'ExperNature',
        '情感支持度': 'ExperSupport',
        '隐私保护': 'ExperPrivacy',
        '个性化程度': 'ExperPersonal',
        '您认为情感陪伴AI以下方面的重要程度如何？—对话自然度': 'ImporNature',
        '情感支持度.1': 'ImporSupport',
        '隐私保护.1': 'ImporPrivacy',
        '个性化程度.1': 'ImporPersonal',
        '您使用情感陪伴AI的主要场景？(睡前放松)': 'ScenarioSleep',
        '11(工作学习间隙)': 'ScenarioWork',
        '11(情绪低落时)': 'ScenarioDown',
        '11(日常娱乐)': 'ScenarioNovel',
        '11(长途旅行中)': 'ScenarioTravel',
        '11(其它)': 'ScenarioOther',
        '您未使用过情感陪伴 AI，原因是?（多选）(不了解产品)': 'NotReasonProduct',
        '12(觉得没必要)': 'NotReasonNece',
        '12(担心隐私问题)': 'NotReasonPrivacy',
        '12(其他（请说明）)': 'NotReasonOther',
        ' 您对尝试使用情感陪伴AI的意愿如何?': 'Try',
        '您对情感陪伴 AI 可能带来的影响有哪些担忧?（多选）(影响现实社交关系)': 'ConcernSocial',
        '14(引发伦理或道德问题)': 'ConcernMoral',
        '14(存在数据隐私和安全风险)': 'ConcernPrivacy',
        '14(使用体验不佳)': 'ConcernExper',
        '14(没有特别担忧)': 'ConcernNot',
        '14(其他（请填写）)': 'ConcernOther',
        '您对情感陪伴AI的付费意愿？': 'Pay',
        '如果按月付款，您期望的情感陪伴AI服务的价格范围是多少?': 'Price',
        '如果愿意付费，您更倾向于哪种付费模式?（单选）': 'PayModel',
        '在选择情感陪伴AI产品时，哪些因素对您最重要?(对话质量)': 'ChooseDialogue',
        '18(价格)': 'ChoosePrice',
        '18(隐私保护)': 'ChoosePrivacy',
        '18(个性化定制)': 'ChoosePersonal',
        '18(品牌信誉)': 'ChooseBrand',
        '18(用户口碑)': 'ChooseUser',
        '18(功能丰富性)': 'ChooseFunction',
        '18(其他（请说明）)': 'ChooseOther',
        '您认为 AI 陪伴的市场前景如何?': 'Market'
    }
    df = df.rename(columns=rename_dict)
    #print(df.columns)
    
    dfuf = df.copy()
    dfkm = df.copy()
    
    ''' ----------------------------------'''
    ''' 所有类别变量做独热编码 '''
    
    ohe = OneHotEncoder(drop='first', sparse_output=False)
    categorical_features = ['Gender','Identity','Support','UsedFreq','PayModel']
    
    #print(df[categorical_features].dtypes)
    
    # 转换数据类型
    for col in categorical_features:
        if df[col].dtype != 'object':
            df[col] = df[col].astype(str)
    
    df_encoded = pd.DataFrame(ohe.fit_transform(df[categorical_features]))
    df_encoded.columns = ohe.get_feature_names_out(categorical_features)
    
    df = df.drop(columns=categorical_features)
    df = pd.concat([df, df_encoded], axis=1)
    
    ohe_dict = {
    'Gender_2': 'Gender_Female',
    'Identity_2': 'Identity_graduate',
    'Identity_3': 'Identity_part',
    'Identity_4': 'Identity_work',
    'Support_2': 'Support_network',
    'Support_3': 'Support_self',
    'Support_4': 'Support_friend',
    'UsedFreq_2': 'UsedFreq_sometimes',
    'UsedFreq_3': 'UsedFreq_heard',
    'UsedFreq_4': 'UsedFreq_never',
    'PayModel_2': 'PayModel_use',
    'PayModel_3': 'PayModel_subs',
    'PayModel_4': 'PayModel_purchase',
    'PayModel_5': 'PayModel_other'
    }
    
    df = df.rename(columns = ohe_dict)
    
    
    ''' ----------------------------------'''
    ''' 填充空值 '''
    dfnn = df.copy()
    fill_scale_0 = [
        "ReasonLonely", "ReasonEmo", "ReasonSupport", "ReasonSocial", "ReasonEnter", 
        "ReasonNovel", "ReasonOther", "ExperOverall", "ExperNature", "ExperSupport", 
        "ExperPrivacy", "ExperPersonal", "ImporNature", "ImporSupport", "ImporPrivacy", 
        "ImporPersonal", "ScenarioSleep", "ScenarioWork", "ScenarioDown", "ScenarioNovel", 
        "ScenarioTravel", "ScenarioOther"
        ]
    dfnn[fill_scale_0] = dfnn[fill_scale_0].fillna(0)

    fill_scale_6 = [
        "NotReasonProduct", "NotReasonNece", "NotReasonPrivacy", "NotReasonOther", "Try"
        ]
    dfnn[fill_scale_6] = dfnn[fill_scale_6].fillna(6)
    

    clean_path = os.path.join(output_path,"cleaned_excel_file.xlsx")
    dfnn.to_excel(clean_path, index=False)
    print(f"数据清洗结果已保存到 {clean_path}")
    
    
    
    ''' ----------------------------------'''
    ''' 除了UsedFreq，其他做独热编码，并填充空值 '''
    ohe = OneHotEncoder(drop='first', sparse_output=False)
    categorical_features = ['Gender','Identity','Support','PayModel']
    
    # 转换数据类型
    for col in categorical_features:
        if dfuf[col].dtype != 'object':
            dfuf[col] = dfuf[col].astype(str)
    
    dfuf_encoded = pd.DataFrame(ohe.fit_transform(dfuf[categorical_features]))
    dfuf_encoded.columns = ohe.get_feature_names_out(categorical_features)
    dfuf = dfuf.drop(columns=categorical_features)
    dfuf = pd.concat([dfuf, dfuf_encoded],axis = 1)
    
    ohe_dict = {
    'Gender_2': 'Gender_Female',
    'Identity_2': 'Identity_graduate',
    'Identity_3': 'Identity_part',
    'Identity_4': 'Identity_work',
    'Support_2': 'Support_network',
    'Support_3': 'Support_self',
    'Support_4': 'Support_friend',
    'PayModel_2': 'PayModel_use',
    'PayModel_3': 'PayModel_subs',
    'PayModel_4': 'PayModel_purchase',
    'PayModel_5': 'PayModel_other'
    }
    
    dfuf = dfuf.rename(columns = ohe_dict)
    dfuf['UsedFreq'] = dfuf['UsedFreq'].replace({1: 4, 2: 3, 3: 2, 4: 1})
    
    fill_scale_0 = [
        "ReasonLonely", "ReasonEmo", "ReasonSupport", "ReasonSocial", "ReasonEnter", 
        "ReasonNovel", "ReasonOther", "ExperOverall", "ExperNature", "ExperSupport", 
        "ExperPrivacy", "ExperPersonal", "ImporNature", "ImporSupport", "ImporPrivacy", 
        "ImporPersonal", "ScenarioSleep", "ScenarioWork", "ScenarioDown", "ScenarioNovel", 
        "ScenarioTravel", "ScenarioOther"
        ]
    dfuf[fill_scale_0] = dfuf[fill_scale_0].fillna(0)

    fill_scale_6 = [
        "NotReasonProduct", "NotReasonNece", "NotReasonPrivacy", "NotReasonOther", "Try"
        ]
    dfuf[fill_scale_6] = dfuf[fill_scale_6].fillna(6)
    
    dfuf.to_excel(os.path.join(output_path,"UsedFreq_cleaned_excel_file.xlsx"), index = False)
    
    
    
    ''' ----------------------------------'''
    ''' 对于Kmeans聚类，独热编码不drop一类，取值标准化 '''
    
    ''' 独热编码 '''
    ohe = OneHotEncoder(drop=None, sparse_output=False)
    categorical_features = ['Gender', 'Identity', 'Support', 'UsedFreq', 'PayModel']
    
    # 确保全是字符串
    for col in categorical_features:
        if dfkm[col].dtype != 'object':
            dfkm[col] = dfkm[col].astype(str)
    
    dfkm_encoded = pd.DataFrame(ohe.fit_transform(dfkm[categorical_features]))
    dfkm_encoded.columns = ohe.get_feature_names_out(categorical_features)
    
    dfkm = dfkm.drop(columns=categorical_features)
    dfkm = pd.concat([dfkm, dfkm_encoded], axis=1)
    
    ohe_dict = {
        'Gender_1': 'Gender_Male',
        'Gender_2': 'Gender_Female',
        'Identity_1': 'Identity_undergraduate',
        'Identity_2': 'Identity_graduate',
        'Identity_3': 'Identity_part',
        'Identity_4': 'Identity_work',
        'Support_1': 'Support_ai',
        'Support_2': 'Support_network',
        'Support_3': 'Support_self',
        'Support_4': 'Support_friend',
        'UsedFreq_1': 'UsedFreq_often',
        'UsedFreq_2': 'UsedFreq_sometimes',
        'UsedFreq_3': 'UsedFreq_heard',
        'UsedFreq_4': 'UsedFreq_never',
        'PayModel_1': 'PayModel_advertisement',
        'PayModel_2': 'PayModel_use',
        'PayModel_3': 'PayModel_subs',
        'PayModel_4': 'PayModel_purchase',
        'PayModel_5': 'PayModel_other'
        }
    
    dfkm = dfkm.rename(columns = ohe_dict)

    
    ''' 填充空值 '''
    fill_scale_0 = [
        "ReasonLonely", "ReasonEmo", "ReasonSupport", "ReasonSocial", "ReasonEnter", 
        "ReasonNovel", "ReasonOther", "ExperOverall", "ExperNature", "ExperSupport", 
        "ExperPrivacy", "ExperPersonal", "ImporNature", "ImporSupport", "ImporPrivacy", 
        "ImporPersonal", "ScenarioSleep", "ScenarioWork", "ScenarioDown", "ScenarioNovel", 
        "ScenarioTravel", "ScenarioOther"
    ]
    dfkm[fill_scale_0] = dfkm[fill_scale_0].fillna(0)
    
    fill_scale_6 = [
        "NotReasonProduct", "NotReasonNece", "NotReasonPrivacy", "NotReasonOther", "Try"
    ]
    dfkm[fill_scale_6] = dfkm[fill_scale_6].fillna(6)
    
    ''' 全部变量标准化 (0–1) '''
    scaler = MinMaxScaler()
    dfkm = pd.DataFrame(scaler.fit_transform(dfkm), columns=dfkm.columns)
    
    dfkm.to_excel(os.path.join(output_path,"kmeansdata_excel_file.xlsx"), index = False)

    
    
''' ----------------------------------'''
''' 因子分析  '''
# 选择用于因子分析的列（数值型变量，如 Likert 量表）
columns_for_factor_analysis = [
    "Loneliness", "Depre", "ReasonLonely", "ReasonEmo", "ReasonSupport", "ReasonSocial", 
    "ReasonEnter", "ReasonNovel", "ReasonOther", "ExperOverall", "ExperNature", "ExperSupport", 
    "ExperPrivacy", "ExperPersonal", "ImporNature", "ImporSupport", "ImporPrivacy", "ImporPersonal",
    "ScenarioSleep", "ScenarioWork", "ScenarioDown", "ScenarioNovel", "ScenarioTravel", "ScenarioOther",
    "NotReasonProduct", "NotReasonNece", "NotReasonPrivacy", "NotReasonOther", "Try", "ConcernSocial", 
    "ConcernMoral", "ConcernPrivacy", "ConcernExper", "ConcernNot", "ConcernOther", "Pay", "Price", 
    "ChooseDialogue", "ChoosePrice", "ChoosePrivacy", "ChoosePersonal", "ChooseBrand", "ChooseUser",
    "ChooseFunction", "ChooseOther", "Market","Support_network", "Support_self", "Support_friend", 
    "UsedFreq_sometimes", "UsedFreq_heard", "UsedFreq_never", "PayModel_use", "PayModel_subs", 
    "PayModel_purchase", "PayModel_other",
]

df_fa = dfnn[columns_for_factor_analysis]

'''
# 删除方差为0的列
df_fa = df_fa.loc[:, df_fa.var() != 0]
# 删除重复列
df_fa = df_fa.loc[:, ~df_fa.T.duplicated()]
# 删除含缺失值的行
df_fa = df_fa.dropna()

print(df_fa)

# 打印含有 NaN 或 Inf 的行
print("包含 NaN 或 Inf 的行：")
# 检查是否有缺失值或无穷大值
invalid_rows = df_fa[df_fa.isnull().any(axis=1) | (df_fa.isin([np.inf, -np.inf]).any(axis=1))]
print(invalid_rows)

# 打印每列的数据类型
print("每列的数据类型：")
print(df_fa.dtypes)
'''

from factor_analyzer import calculate_kmo

# KMO 检验
kmo_all, kmo_model = calculate_kmo(df_fa)
print(f"KMO 值: {kmo_model:.3f}")

from scipy.stats import chi2

def bartlett_sphericity_manual(data):
    """
    计算 Bartlett 球形检验
    """
    n, p = data.shape
    corr_matrix = np.corrcoef(data, rowvar=False) 
    det_corr_matrix = np.linalg.det(corr_matrix)

    chi_square_value = -(n - 1 - (2 * p + 5) / 6) * np.log(det_corr_matrix)
    df = p * (p - 1) / 2  # 自由度
    p_value = 1 - chi2.cdf(chi_square_value, df)

    return chi_square_value, p_value

chi_square_value, p_value = bartlett_sphericity_manual(df_fa.to_numpy())
print(f"Bartlett's Test p值: {p_value:.5f}")

log_path = os.path.join(output_path, "因子分析的检验.txt")
with open(log_path, 'w', encoding='utf-8') as f:
    f.write(f"KMO 值: {kmo_model:.3f}\n")
    f.write(f"Bartlett's Test p值: {p_value:.5f}\n")

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fa = FactorAnalyzer(n_factors=len(df_fa.columns), rotation=None)
fa.fit(df_fa)

# 计算特征值
ev, v = fa.get_eigenvalues()

# 画碎石图
plt.figure(figsize=(8,5))
plt.scatter(range(1, len(df_fa.columns)+1), ev)
plt.plot(range(1, len(df_fa.columns)+1), ev)
plt.xlabel("因子数")
plt.ylabel("特征值")
plt.title("all_碎石图")
plt.grid()
plt.show()

'''
num_factors = 3
'''

while True:
    try:
        num_factors = int(input("根据碎石图，请输入你希望提取的因子数量（整数）："))
        if 1 <= num_factors <= len(df_fa.columns):
            break
        else:
            print(f"请输入 1 到 {len(df_fa.columns)} 之间的整数。")
    except ValueError:
        print("请输入有效整数")


fa = FactorAnalyzer(n_factors=num_factors, rotation="varimax")  # 使用 Varimax 旋转
fa.fit(df_fa)

# 因子载荷矩阵
loadings = pd.DataFrame(fa.loadings_, index=columns_for_factor_analysis)

# 归类：找出载荷最大且 > 0.4 的因子
factor_assignments = []
for var, row in loadings.iterrows():
    max_loading = row.max()
    if max_loading > 0.4:
        factor_idx = row.idxmax()  # 找到最大载荷的因子索引
        factor_assignments.append([var, f'Factor {factor_idx + 1}', max_loading])

factor_df = pd.DataFrame(factor_assignments, columns=['Variable', 'Factor', 'Loading'])


factor_df.to_excel(os.path.join(output_path,"all_factor_assignments.xlsx"), index=False)

#print(factor_df)

    


''' Emo列生成（根据因子分析结果） '''

from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import MinMaxScaler

def generate_emo_column(df, var1='Loneliness', var2='Depre', emo_col='Emo'):
    # 判断两个变量是否都存在
    if var1 not in df.columns or var2 not in df.columns:
        print(f"缺少变量 {var1} 或 {var2}，跳过处理。")
        return df
    
    corr = df[[var1, var2]].corr().iloc[0, 1]
    print(f"{var1} 和 {var2} 的相关系数为: {corr:.3f}")
    
    if abs(corr) >= 0.6:
        # 因子分析生成 Emo
        fa = FactorAnalysis(n_components=1)
        emo_scores = fa.fit_transform(df[[var1, var2]])
        df[emo_col] = emo_scores.flatten()
        scaler = MinMaxScaler(feature_range=(1, 5))
        df[emo_col] = scaler.fit_transform(df[[emo_col]])

        print(f"{emo_col}列使用因子分析生成")
    else:
        # 直接平均生成 Emo
        df[emo_col] = df[[var1, var2]].mean(axis=1)
        print(f"{emo_col}列使用平均值生成")
    
    return df

dfnn = generate_emo_column(dfnn)
dfkm = generate_emo_column(dfkm)
dfuf = generate_emo_column(dfuf)
df = generate_emo_column(df)




''' 回归分析 '''

stata.run('display "hello from stata"')

stata.pdataframe_to_data(dfnn, force = True)

stata.run(f'cd "{output_path}"')

stata.run('asdoc mvreg ImporNature ImporSupport ImporPrivacy ImporPersonal = Age Spending ReasonLonely ReasonEmo ReasonSupport ReasonSocial ReasonEnter ReasonNovel ExperOverall ExperNature ExperSupport ExperPrivacy ExperPersonal ScenarioSleep ScenarioWork ScenarioDown ScenarioNovel ScenarioTravel Pay Price Gender_Female Identity_graduate Identity_part Identity_work Support_network Support_self Support_friend UsedFreq_sometimes Emo, replace save(功能重要性_回归.doc)') 

stata.run('regress ImporPersonal Age Spending ReasonLonely ReasonEmo ReasonSupport ReasonSocial ReasonEnter ReasonNovel ExperOverall ExperNature ExperSupport ExperPrivacy ExperPersonal ScenarioSleep ScenarioWork ScenarioDown ScenarioNovel ScenarioTravel Pay Price Gender_Female Identity_graduate Identity_part Identity_work Support_network Support_self Support_friend UsedFreq_sometimes Emo')
stata.run('predict residuals1, residuals')

stata.run('log using 功能需求分析_回归检验.txt, replace text')
stata.run('swilk residuals1')
stata.run('estat hettest')
stata.run('log close')


stata.run('regress ImporNature Age Spending ReasonLonely ReasonEmo ReasonSupport ReasonSocial ReasonEnter ReasonNovel ExperOverall ExperNature ExperSupport ExperPrivacy ExperPersonal ScenarioSleep ScenarioWork ScenarioDown ScenarioNovel ScenarioTravel Pay Price Gender_Female Identity_graduate Identity_part Identity_work Support_network Support_self Support_friend UsedFreq_sometimes Emo')
res = stata.get_ereturn()  # 获取回归结果字典&#8203;:contentReference[oaicite:1]{index=1}

# 2. 提取系数和协方差矩阵
# e(b) 是 1×(p+1) 的系数行向量，包含常数项
coef_array = np.array(res['e(b)']).flatten()  # 将形如[[b0,b1,...]]拉平
# e(V) 是 (p+1)×(p+1) 的协方差矩阵
cov_matrix = np.array(res['e(V)'])
# 获取自变量名列表（假设已知或通过脚本维护），如：
var_names = ['Age', 'Spending', 'ReasonLonely', 'ReasonEmo', 'ReasonSupport',
        'ReasonSocial', 'ReasonEnter', 'ReasonNovel', 'ExperOverall', 'ExperNature',
        'ExperSupport', 'ExperPrivacy', 'ExperPersonal', 'ScenarioSleep', 'ScenarioWork',
        'ScenarioDown', 'ScenarioNovel', 'ScenarioTravel', 'Pay', 'Price',
        'Gender_Female', 'Identity_graduate', 'Identity_part', 'Identity_work',
        'Support_network', 'Support_self', 'Support_friend', 'UsedFreq_sometimes', 'Emo','constant'
        ]  # 对应 e(b) 中的系数顺序
# 注意：索引 0 对应常数项，后续依序对应 x1,x2,...

# 3. 计算统计量：标准误、t值、p值
ses = np.sqrt(np.diag(cov_matrix))    # 系数标准误
t_stats = coef_array / ses           # t 值
df_resid = res['e(df_r)']            # 残差自由度
p_vals = 2 * stats.t.sf(np.abs(t_stats), df_resid)  # 双尾 t 检验 p 值

# 4. 筛选显著变量（不含常数项）
sig_mask = (p_vals < 0.1) & (np.array(var_names) != 'Constant')
sig_idx = np.where(sig_mask)[0]
# 提取显著变量的名称、系数和 p 值
sig_names = [var_names[i] for i in sig_idx]
sig_coefs = coef_array[sig_idx]
sig_pvals = p_vals[sig_idx]
# 按系数大小排序（这里按从大到小）
order = np.argsort(sig_coefs)[::-1]
sig_names = [sig_names[i] for i in order]
sig_coefs = sig_coefs[order]
sig_pvals = sig_pvals[order]

# 5. 绘制条形图和散点图
plt.rcParams['font.sans-serif'] = ['SimHei']      # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False       # 正确显示负号
fig, ax = plt.subplots(figsize=(6,4))
y_pos = np.arange(len(sig_coefs))

# 根据系数正负设置条形颜色
colors = ['#fdd5b1' if c > 0 else '#f88379' for c in sig_coefs]
# 绘制水平条形图
ax.barh(y_pos, sig_coefs, color=colors, edgecolor='gray')
ax.set_yticks(y_pos)
ax.set_yticklabels(sig_names, rotation=30, ha='right')  # 变量名倾斜30°
ax.set_xlabel("回归系数")
ax.set_title("功能需求分析的线性回归结果（以对话自然度为例）")


# 添加第二坐标轴用于 p 值散点
ax2 = ax.twiny()
ax2.set_xlim(-0.05, 1)  
# 绘制 p 值散点
ax2.scatter(sig_pvals, y_pos, color='#00555A', marker='o', s=50, label="p 值")
ax2.set_xlabel("P 值")

# 图例
pos_patch = mpatches.Patch(color='#fdd5b1', label='正向系数')
neg_patch = mpatches.Patch(color='#f88379', label='负向系数')
ax.legend(handles=[pos_patch, neg_patch, ax2.collections[0]], loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(output_path,'功能需求分析的线性回归结果（对话自然度）'), dpi = 300)
#plt.show()

''' -------------------------------------------------------- '''


stata.run('regress ImporSupport Age Spending ReasonLonely ReasonEmo ReasonSupport ReasonSocial ReasonEnter ReasonNovel ExperOverall ExperNature ExperSupport ExperPrivacy ExperPersonal ScenarioSleep ScenarioWork ScenarioDown ScenarioNovel ScenarioTravel Pay Price Gender_Female Identity_graduate Identity_part Identity_work Support_network Support_self Support_friend UsedFreq_sometimes Emo')
res = stata.get_ereturn()  # 获取回归结果字典&#8203;:contentReference[oaicite:1]{index=1}

# 2. 提取系数和协方差矩阵
# e(b) 是 1×(p+1) 的系数行向量，包含常数项
coef_array = np.array(res['e(b)']).flatten()  # 将形如[[b0,b1,...]]拉平
# e(V) 是 (p+1)×(p+1) 的协方差矩阵
cov_matrix = np.array(res['e(V)'])
# 获取自变量名列表（假设已知或通过脚本维护），如：
var_names = ['Age', 'Spending', 'ReasonLonely', 'ReasonEmo', 'ReasonSupport',
        'ReasonSocial', 'ReasonEnter', 'ReasonNovel', 'ExperOverall', 'ExperNature',
        'ExperSupport', 'ExperPrivacy', 'ExperPersonal', 'ScenarioSleep', 'ScenarioWork',
        'ScenarioDown', 'ScenarioNovel', 'ScenarioTravel', 'Pay', 'Price',
        'Gender_Female', 'Identity_graduate', 'Identity_part', 'Identity_work',
        'Support_network', 'Support_self', 'Support_friend', 'UsedFreq_sometimes', 'Emo','constant'
        ]  # 对应 e(b) 中的系数顺序
# 注意：索引 0 对应常数项，后续依序对应 x1,x2,...

# 3. 计算统计量：标准误、t值、p值
ses = np.sqrt(np.diag(cov_matrix))    # 系数标准误
t_stats = coef_array / ses           # t 值
df_resid = res['e(df_r)']            # 残差自由度
p_vals = 2 * stats.t.sf(np.abs(t_stats), df_resid)  # 双尾 t 检验 p 值

# 4. 筛选显著变量（不含常数项）
sig_mask = (p_vals < 0.1) & (np.array(var_names) != 'Constant')
sig_idx = np.where(sig_mask)[0]
# 提取显著变量的名称、系数和 p 值
sig_names = [var_names[i] for i in sig_idx]
sig_coefs = coef_array[sig_idx]
sig_pvals = p_vals[sig_idx]
# 按系数大小排序（这里按从大到小）
order = np.argsort(sig_coefs)[::-1]
sig_names = [sig_names[i] for i in order]
sig_coefs = sig_coefs[order]
sig_pvals = sig_pvals[order]

# 5. 绘制条形图和散点图
plt.rcParams['font.sans-serif'] = ['SimHei']      # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False       # 正确显示负号
fig, ax = plt.subplots(figsize=(6,4))
y_pos = np.arange(len(sig_coefs))

# 根据系数正负设置条形颜色
colors = ['#fdd5b1' if c > 0 else '#f88379' for c in sig_coefs]
# 绘制水平条形图
ax.barh(y_pos, sig_coefs, color=colors, edgecolor='gray')
ax.set_yticks(y_pos)
ax.set_yticklabels(sig_names, rotation=30, ha='right')  # 变量名倾斜30°
ax.set_xlabel("回归系数")
ax.set_title("功能需求分析的线性回归结果（以情感支持为例）")

# 添加第二坐标轴用于 p 值散点
ax2 = ax.twiny()
ax2.set_xlim(-0.05, 1)  
# 绘制 p 值散点
ax2.scatter(sig_pvals, y_pos, color='#00555A', marker='o', s=50, label="p 值")
ax2.set_xlabel("P 值")

# 图例
pos_patch = mpatches.Patch(color='#fdd5b1', label='正向系数')
neg_patch = mpatches.Patch(color='#f88379', label='负向系数')
ax.legend(handles=[pos_patch, neg_patch, ax2.collections[0]], loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(output_path,'功能需求分析的线性回归结果（情感支持）'), dpi = 300)
#plt.show()


''' -------------------------------------------------------- '''


stata.run('regress ImporPrivacy Age Spending ReasonLonely ReasonEmo ReasonSupport ReasonSocial ReasonEnter ReasonNovel ExperOverall ExperNature ExperSupport ExperPrivacy ExperPersonal ScenarioSleep ScenarioWork ScenarioDown ScenarioNovel ScenarioTravel Pay Price Gender_Female Identity_graduate Identity_part Identity_work Support_network Support_self Support_friend UsedFreq_sometimes Emo')
res = stata.get_ereturn()  # 获取回归结果字典&#8203;:contentReference[oaicite:1]{index=1}

# 2. 提取系数和协方差矩阵
# e(b) 是 1×(p+1) 的系数行向量，包含常数项
coef_array = np.array(res['e(b)']).flatten()  # 将形如[[b0,b1,...]]拉平
# e(V) 是 (p+1)×(p+1) 的协方差矩阵
cov_matrix = np.array(res['e(V)'])
# 获取自变量名列表（假设已知或通过脚本维护），如：
var_names = ['Age', 'Spending', 'ReasonLonely', 'ReasonEmo', 'ReasonSupport',
        'ReasonSocial', 'ReasonEnter', 'ReasonNovel', 'ExperOverall', 'ExperNature',
        'ExperSupport', 'ExperPrivacy', 'ExperPersonal', 'ScenarioSleep', 'ScenarioWork',
        'ScenarioDown', 'ScenarioNovel', 'ScenarioTravel', 'Pay', 'Price',
        'Gender_Female', 'Identity_graduate', 'Identity_part', 'Identity_work',
        'Support_network', 'Support_self', 'Support_friend', 'UsedFreq_sometimes', 'Emo','constant'
        ]  # 对应 e(b) 中的系数顺序
# 注意：索引 0 对应常数项，后续依序对应 x1,x2,...

# 3. 计算统计量：标准误、t值、p值
ses = np.sqrt(np.diag(cov_matrix))    # 系数标准误
t_stats = coef_array / ses           # t 值
df_resid = res['e(df_r)']            # 残差自由度
p_vals = 2 * stats.t.sf(np.abs(t_stats), df_resid)  # 双尾 t 检验 p 值

# 4. 筛选显著变量（不含常数项）
sig_mask = (p_vals < 0.1) & (np.array(var_names) != 'Constant')
sig_idx = np.where(sig_mask)[0]
# 提取显著变量的名称、系数和 p 值
sig_names = [var_names[i] for i in sig_idx]
sig_coefs = coef_array[sig_idx]
sig_pvals = p_vals[sig_idx]
# 按系数大小排序（这里按从大到小）
order = np.argsort(sig_coefs)[::-1]
sig_names = [sig_names[i] for i in order]
sig_coefs = sig_coefs[order]
sig_pvals = sig_pvals[order]

# 5. 绘制条形图和散点图
plt.rcParams['font.sans-serif'] = ['SimHei']      # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False       # 正确显示负号
fig, ax = plt.subplots(figsize=(6,4))
y_pos = np.arange(len(sig_coefs))

# 根据系数正负设置条形颜色
colors = ['#fdd5b1' if c > 0 else '#f88379' for c in sig_coefs]
# 绘制水平条形图
ax.barh(y_pos, sig_coefs, color=colors, edgecolor='gray')
ax.set_yticks(y_pos)
ax.set_yticklabels(sig_names, rotation=30, ha='right')  # 变量名倾斜30°
ax.set_xlabel("回归系数")
ax.set_title("功能需求分析的线性回归结果（以隐私保护为例）")

# 添加第二坐标轴用于 p 值散点
ax2 = ax.twiny()
ax2.set_xlim(-0.05, 1)  
# 绘制 p 值散点
ax2.scatter(sig_pvals, y_pos, color='#00555A', marker='o', s=50, label="p 值")
ax2.set_xlabel("P 值")

# 图例
pos_patch = mpatches.Patch(color='#fdd5b1', label='正向系数')
neg_patch = mpatches.Patch(color='#f88379', label='负向系数')
ax.legend(handles=[pos_patch, neg_patch, ax2.collections[0]], loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(output_path,'功能需求分析的线性回归结果（隐私保护）'), dpi = 300)
#plt.show()    


''' -------------------------------------------------------- '''

stata.run('regress ImporPersonal Age Spending ReasonLonely ReasonEmo ReasonSupport ReasonSocial ReasonEnter ReasonNovel ExperOverall ExperNature ExperSupport ExperPrivacy ExperPersonal ScenarioSleep ScenarioWork ScenarioDown ScenarioNovel ScenarioTravel Pay Price Gender_Female Identity_graduate Identity_part Identity_work Support_network Support_self Support_friend UsedFreq_sometimes Emo')
res = stata.get_ereturn()  # 获取回归结果字典&#8203;:contentReference[oaicite:1]{index=1}

# 2. 提取系数和协方差矩阵
# e(b) 是 1×(p+1) 的系数行向量，包含常数项
coef_array = np.array(res['e(b)']).flatten()  # 将形如[[b0,b1,...]]拉平
# e(V) 是 (p+1)×(p+1) 的协方差矩阵
cov_matrix = np.array(res['e(V)'])
# 获取自变量名列表（假设已知或通过脚本维护），如：
var_names = ['Age', 'Spending', 'ReasonLonely', 'ReasonEmo', 'ReasonSupport',
        'ReasonSocial', 'ReasonEnter', 'ReasonNovel', 'ExperOverall', 'ExperNature',
        'ExperSupport', 'ExperPrivacy', 'ExperPersonal', 'ScenarioSleep', 'ScenarioWork',
        'ScenarioDown', 'ScenarioNovel', 'ScenarioTravel', 'Pay', 'Price',
        'Gender_Female', 'Identity_graduate', 'Identity_part', 'Identity_work',
        'Support_network', 'Support_self', 'Support_friend', 'UsedFreq_sometimes', 'Emo', 'constant'
        ]  # 对应 e(b) 中的系数顺序
# 注意：索引 0 对应常数项，后续依序对应 x1,x2,...

# 3. 计算统计量：标准误、t值、p值
ses = np.sqrt(np.diag(cov_matrix))    # 系数标准误
t_stats = coef_array / ses           # t 值
df_resid = res['e(df_r)']            # 残差自由度
p_vals = 2 * stats.t.sf(np.abs(t_stats), df_resid)  # 双尾 t 检验 p 值

# 4. 筛选显著变量（不含常数项）
sig_mask = (p_vals < 0.1) & (np.array(var_names) != 'Constant')
sig_idx = np.where(sig_mask)[0]
# 提取显著变量的名称、系数和 p 值
sig_names = [var_names[i] for i in sig_idx]
sig_coefs = coef_array[sig_idx]
sig_pvals = p_vals[sig_idx]
# 按系数大小排序（这里按从大到小）
order = np.argsort(sig_coefs)[::-1]
sig_names = [sig_names[i] for i in order]
sig_coefs = sig_coefs[order]
sig_pvals = sig_pvals[order]

# 5. 绘制条形图和散点图
plt.rcParams['font.sans-serif'] = ['SimHei']      # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False       # 正确显示负号
fig, ax = plt.subplots(figsize=(6,4))
y_pos = np.arange(len(sig_coefs))

# 根据系数正负设置条形颜色
colors = ['#fdd5b1' if c > 0 else '#f88379' for c in sig_coefs]
# 绘制水平条形图
ax.barh(y_pos, sig_coefs, color=colors, edgecolor='gray')
ax.set_yticks(y_pos)
ax.set_yticklabels(sig_names, rotation=30, ha='right')  # 变量名倾斜30°
ax.set_xlabel("回归系数")
ax.set_title("功能需求分析的线性回归结果（以个性化定制为例）")

# 添加第二坐标轴用于 p 值散点
ax2 = ax.twiny()
ax2.set_xlim(-0.05, 1)  
# 绘制 p 值散点
ax2.scatter(sig_pvals, y_pos, color='#00555A', marker='o', s=50, label="p 值")
ax2.set_xlabel("P 值")

pos_patch = mpatches.Patch(color='#fdd5b1', label='正向系数')
neg_patch = mpatches.Patch(color='#f88379', label='负向系数')
ax.legend(handles=[pos_patch, neg_patch, ax2.collections[0]], loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(output_path,'功能需求分析的线性回归结果（个性化定制功能）'), dpi = 300)
#plt.show()


''' -------------------------------------------------------- '''

''' 整体体验的回归 '''

stata.pdataframe_to_data(dfnn, force=True)
stata.run('asdoc ologit ExperOverall ExperNature ExperSupport ExperPrivacy ExperPersonal,  replace save(整体体验_回归.doc)')
res = stata.get_ereturn()

b = res['e(b)'].flatten()
V = res['e(V)']

variables = ["ExperNature", "ExperSupport", "ExperPrivacy", "ExperPersonal"]
coefs = b[:len(variables)]
se = np.sqrt(np.diag(V))[:len(variables)]
t_vals = coefs / se
p_vals = 2 * (1 - norm.cdf(np.abs(t_vals)))

# 构建结果表并筛选显著变量
results_df = pd.DataFrame({
    'coef': coefs,
    'se': se,
    't': t_vals,
    'p': p_vals
}, index=variables)
sig_df = results_df[results_df['p'] < 0.1]
sig_df = sig_df.sort_values('coef', ascending=False)

# 可视化设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(6, 4))
y_pos = np.arange(len(sig_df))

# 条形颜色
colors = ['#fdd5b1' if c > 0 else '#f88379' for c in sig_df['coef']]
ax.barh(y_pos, sig_df['coef'], color=colors, edgecolor='gray')
ax.set_yticks(y_pos)
ax.set_yticklabels(sig_df.index, rotation=30, ha='right')
ax.set_xlabel("回归系数")
ax.set_title("整体体验的有序回归结果")

# 添加双坐标轴绘制 p 值
ax2 = ax.twiny()
ax2.set_xlim(-0.05, 1)
ax2.scatter(sig_df['p'], y_pos, color='#00555A', marker='o', s=50)
ax2.set_xlabel("P 值")

# 图例设置
pos_patch = mpatches.Patch(color='#fdd5b1', label='正向系数')
neg_patch = mpatches.Patch(color='#f88379', label='负向系数')
pval_patch = mpatches.Patch(color='#00555A', label='p值')
ax.legend(handles=[pos_patch, neg_patch, pval_patch], loc='upper right')

# 保存并展示
plt.tight_layout()
plt.savefig("C:/Users/Lucius/Desktop/output/整体体验_回归结果.png", dpi=300)
#plt.show()

stata.run('regress ExperOverall ExperNature ExperSupport ExperPrivacy ExperPersonal')
stata.run('log using 整体体验_回归检验.txt, replace text')
stata.run('vif')
stata.run('log close')


''' -------------------------------------------------------- '''

''' 付费意愿的回归 '''

stata.pdataframe_to_data(dfnn, force=True)
stata.run(f'cd "{output_path}"')

stata.run('asdoc regress Pay Age Spending Loneliness Depre UsedFreq_sometimes UsedFreq_heard UsedFreq_never ReasonLonely ReasonEmo ReasonSupport ReasonSocial ReasonEnter ReasonNovel ExperOverall ExperNature ExperSupport ExperPrivacy ExperPersonal ScenarioSleep ScenarioWork ScenarioDown ScenarioNovel ScenarioTravel NotReasonProduct NotReasonNece NotReasonPrivacy Try Gender_Female Identity_graduate Identity_part Identity_work, replace save(付费意愿_回归.doc)')
 
stata.run('regress Pay Age Spending Loneliness Depre UsedFreq_sometimes UsedFreq_heard UsedFreq_never ReasonLonely ReasonEmo ReasonSupport ReasonSocial ReasonEnter ReasonNovel ExperOverall ExperNature ExperSupport ExperPrivacy ExperPersonal ScenarioSleep ScenarioWork ScenarioDown ScenarioNovel ScenarioTravel NotReasonProduct NotReasonNece NotReasonPrivacy Try Gender_Female Identity_graduate Identity_part Identity_work')
stata.run('predict resid_pay, residual')

stata.run('log using 付费意愿_回归检验.txt, replace text')
stata.run('swilk resid_pay')
stata.run('estat hettest')
stata.run('log close')

# Python 提取回归结果
res = stata.get_ereturn()
coef_array = np.array(res['e(b)']).flatten()
cov_matrix = np.array(res['e(V)'])
df_resid = res['e(df_r)']

var_names = ['Age', 'Spending', 'Loneliness', 'Depre', 'UsedFreq_sometimes', 'UsedFreq_heard','UsedFreq_never', 
             'ReasonLonely', 'ReasonEmo', 'ReasonSupport', 'ReasonSocial', 'ReasonEnter', 'ReasonNovel',
             'ExperOverall', 'ExperNature', 'ExperSupport', 'ExperPrivacy', 'ExperPersonal',
             'ScenarioSleep', 'ScenarioWork', 'ScenarioDown', 'ScenarioNovel', 'ScenarioTravel',
             'NotReasonProduct', 'NotReasonNece', 'NotReasonPrivacy', 'Try', 'Gender_Female',
             'Identity_graduate', 'Identity_part', 'Identity_work','Constant']

ses = np.sqrt(np.diag(cov_matrix))
t_stats = coef_array / ses
p_vals = 2 * stats.t.sf(np.abs(t_stats), df_resid)

sig_mask = (p_vals < 0.1) & (np.array(var_names) != 'Constant')
sig_idx = np.where(sig_mask)[0]
sig_names = [var_names[i] for i in sig_idx]
sig_coefs = coef_array[sig_idx]
sig_pvals = p_vals[sig_idx]

order = np.argsort(sig_coefs)[::-1]
sig_names = [sig_names[i] for i in order]
sig_coefs = sig_coefs[order]
sig_pvals = sig_pvals[order]

# 可视化
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots(figsize=(6, 4))
y_pos = np.arange(len(sig_coefs))
colors = ['#fdd5b1' if c > 0 else '#f88379' for c in sig_coefs]
ax.barh(y_pos, sig_coefs, color=colors, edgecolor='gray')
ax.set_yticks(y_pos)
ax.set_yticklabels(sig_names, rotation=30, ha='right')
ax.set_xlabel("回归系数")
ax.set_title("付费意愿分析的回归结果")

ax2 = ax.twiny()
ax2.set_xlim(-0.05, 1)
ax2.scatter(sig_pvals, y_pos, color='#00555A', marker='o', s=50, label="p 值")
ax2.set_xlabel("P 值")

pos_patch = mpatches.Patch(color='#fdd5b1', label='正向系数')
neg_patch = mpatches.Patch(color='#f88379', label='负向系数')
ax.legend(handles=[pos_patch, neg_patch, ax2.collections[0]], loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(output_path, '付费意愿分析的回归结果.png'), dpi=300)
#plt.show()

''' -------------------------------------------------------- '''

''' 使用频率的回归 '''

stata.pdataframe_to_data(dfuf, force=True)
stata.run('asdoc ologit UsedFreq Age Spending Gender_Female Identity_graduate Identity_part Identity_work Loneliness Depre ReasonLonely ReasonEmo ReasonSupport ReasonSocial ReasonEnter ReasonNovel,  replace save(使用频率_回归.doc)')
res = stata.get_ereturn()

b = res['e(b)'].flatten()
V = res['e(V)']

variables = ['Age', 'Spending', 'Gender_Female', 'Identity_graduate', 'Identity_part', 'Identity_work', 
      'Loneliness', 'Depre', 'ReasonLonely', 'ReasonEmo', 'ReasonSupport', 'ReasonSocial', 
      'ReasonEnter', 'ReasonNovel']
coefs = b[:len(variables)]
se = np.sqrt(np.diag(V))[:len(variables)]
t_vals = coefs / se
p_vals = 2 * (1 - norm.cdf(np.abs(t_vals)))

# 构建结果表并筛选显著变量
results_df = pd.DataFrame({
    'coef': coefs,
    'se': se,
    't': t_vals,
    'p': p_vals
}, index=variables)
sig_df = results_df[results_df['p'] < 0.1]
sig_df = sig_df.sort_values('coef', ascending=False)

# 可视化设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(6, 4))
y_pos = np.arange(len(sig_df))

# 条形颜色
colors = ['#fdd5b1' if c > 0 else '#f88379' for c in sig_df['coef']]
ax.barh(y_pos, sig_df['coef'], color=colors, edgecolor='gray')
ax.set_yticks(y_pos)
ax.set_yticklabels(sig_df.index, rotation=30, ha='right')
ax.set_xlabel("回归系数")
ax.set_title("使用频率的有序回归结果")

# 添加双坐标轴绘制 p 值
ax2 = ax.twiny()
ax2.set_xlim(-0.05, 1)
ax2.scatter(sig_df['p'], y_pos, color='#00555A', marker='o', s=50)
ax2.set_xlabel("P 值")

# 图例设置
pos_patch = mpatches.Patch(color='#fdd5b1', label='正向系数')
neg_patch = mpatches.Patch(color='#f88379', label='负向系数')
pval_patch = mpatches.Patch(color='#00555A', label='p值')
ax.legend(handles=[pos_patch, neg_patch, pval_patch], loc='upper right')

# 保存并展示
plt.tight_layout()
plt.savefig("C:/Users/Lucius/Desktop/output/使用频率_回归结果.png", dpi=300)
#plt.show()

stata.run('regress UsedFreq Age Spending Gender_Female Identity_graduate Identity_part Identity_work Loneliness Depre ReasonLonely ReasonEmo ReasonSupport ReasonSocial ReasonEnter ReasonNovel')
stata.run('log using 使用频率_回归检验.txt, replace text')
stata.run('vif')
stata.run('log close')
    
''' ----------------------------------'''
''' Kmeans聚类分析 '''

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

''' 肘部法 (SSE) '''
K_range = range(1, 11)
sse = []
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(dfkm)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, sse, marker='o', linestyle='-')
plt.xlabel('簇的个数 K')
plt.ylabel('误差平方和 (SSE)')
plt.title('肘部法确定最佳 K 值')
plt.xticks(K_range)
plt.grid()
plt.savefig(os.path.join(output_path, '肘部法_最佳K值.png'))
plt.show()
plt.close()

''' 计算轮廓系数 '''
silhouette_scores = []
K_range = range(2, 11)
for K in K_range:
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels = kmeans.fit_predict(dfkm)
    score = silhouette_score(dfkm, labels)
    silhouette_scores.append(score)

plt.figure(figsize=(8, 5))
plt.plot(K_range, silhouette_scores, marker='o', linestyle='-')
plt.xlabel('聚类数 K')
plt.ylabel('轮廓系数')
plt.title('不同 K 值的轮廓系数')
plt.grid()
plt.savefig(os.path.join(output_path, '轮廓系数_不同K值.png'))
plt.show()
plt.close()

''' 让用户输入K值 '''
while True:
    try:
        user_K = int(input("请根据肘部图和轮廓系数图选择最终聚类个数 K: "))
        if 1 <= user_K <= 10:
            break
        else:
            print("请输入1到10之间的整数。")
    except ValueError:
        print("请输入有效整数")


''' 聚类 '''
kmeans = KMeans(n_clusters=user_K, random_state=42, n_init=10)
dfkm['Cluster'] = kmeans.fit_predict(dfkm)

dfkm.to_excel(os.path.join(output_path, "聚类结果.xlsx"), index=False)

''' 计算并保存聚类中心、样本数量、各簇均值 '''
centroids = kmeans.cluster_centers_
unique, counts = np.unique(kmeans.labels_, return_counts=True)
cluster_sizes = dict(zip(unique, counts))

df_centroids = pd.DataFrame(centroids, columns=[f'Feature_{i+1}' for i in range(centroids.shape[1])])
df_centroids.to_excel(os.path.join(output_path, '聚类中心.xlsx'), index=False)

df_cluster_sizes = pd.DataFrame(list(cluster_sizes.items()), columns=['Cluster', 'Sample_Count'])
df_cluster_sizes.to_excel(os.path.join(output_path, '每个簇的样本数量.xlsx'), index=False)

cluster_means = dfkm.groupby('Cluster').mean(numeric_only=True).round(2).reset_index()
cluster_means.to_excel(os.path.join(output_path, '各簇样本均值.xlsx'), index=False)

''' PCA 2维可视化 '''
pca = PCA(n_components=2)
df_pca = pca.fit_transform(dfkm.drop('Cluster', axis=1))

plt.figure(figsize=(8, 6))
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=dfkm['Cluster'], cmap='viridis', alpha=0.7)
plt.title('KMeans 聚类结果 (PCA_2D降维)')
plt.xlabel('主成分 1')
plt.ylabel('主成分 2')
plt.colorbar(label='Cluster')
plt.savefig(os.path.join(output_path, 'KMeans_PCA_2D.png'))
#plt.show()
plt.close()

''' PCA 3维可视化 '''
pca_3d = PCA(n_components=3)
df_pca_3d = pca_3d.fit_transform(dfkm.drop('Cluster', axis=1))

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(df_pca_3d[:, 0], df_pca_3d[:, 1], df_pca_3d[:, 2], c=dfkm['Cluster'], cmap='viridis', alpha=0.7)
ax.set_title('KMeans 聚类结果 (PCA_3D降维)')
ax.set_xlabel('主成分 1')
ax.set_ylabel('主成分 2')
ax.set_zlabel('主成分 3')
plt.colorbar(sc, label='Cluster')
plt.savefig(os.path.join(output_path, 'KMeans_PCA_3D.png'))
#plt.show()
plt.close()

''' t-SNE 2维可视化 '''
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
df_tsne = tsne.fit_transform(dfkm.drop('Cluster', axis=1))

plt.figure(figsize=(8, 6))
plt.scatter(df_tsne[:, 0], df_tsne[:, 1], c=dfkm['Cluster'], cmap='viridis', alpha=0.7)
plt.title('KMeans 聚类结果 (t-SNE降维)')
plt.xlabel('t-SNE 维度 1')
plt.ylabel('t-SNE 维度 2')
plt.colorbar(label='Cluster')
plt.savefig(os.path.join(output_path, 'KMeans_tSNE_2D.png'))
#plt.show()
plt.close()

''' 聚类中心热力图  '''
centroids_df = pd.DataFrame(centroids, columns=dfkm.drop('Cluster', axis=1).columns)
centroids_df = centroids_df.T

plt.figure(figsize=(1 + user_K, max(6, centroids_df.shape[0]*0.5)))  # 高度随特征数自动调整
sns.heatmap(centroids_df, annot=False, cmap='coolwarm', cbar=True)
plt.title('KMeans 聚类中心热力图')
plt.xlabel('Cluster')
plt.ylabel('特征')
plt.tight_layout()
plt.savefig(os.path.join(output_path, '聚类中心热力图.png'))
#plt.show()
plt.close()


print(f'聚类分析已完成，聚类结果及可视化图形已保存至{output_path}')


''' ----------------------------------'''
''' 结构方程 '''
# 定义路径模型
model_desc = """
# 测量模型
User =~ Age + Spending + Gender_Female + Identity_graduate + Identity_part + Identity_work
Emo =~ Loneliness + Depre 
Reason =~ ReasonLonely + ReasonEmo + ReasonSupport + ReasonSocial + ReasonEnter + ReasonNovel + ReasonOther 
Exper=~ ExperOverall + ExperNature + ExperSupport + ExperPrivacy + ExperPersonal
Impor =~ ImporNature + ImporSupport + ImporPrivacy + ImporPersonal 
Scenario =~ ScenarioSleep + ScenarioWork + ScenarioDown + ScenarioNovel + ScenarioTravel + ScenarioOther 
NotReason =~ NotReasonProduct + NotReasonNece + NotReasonPrivacy + NotReasonOther
Accept =~ Try + ConcernSocial + ConcernMoral + ConcernPrivacy + ConcernExper + ConcernNot + ConcernOther 
WTP =~ Pay + Price


# 结构模型
UsedFreq ~ User + Emo
Impor ~ Reason + Exper + UsedFreq
Accept ~ Exper + Impor + NotReason
WTP ~ Accept
Market ~ UsedFreq + Exper + WTP + Accept + User
"""

# 创建SEM模型
model = Model(model_desc)

# 将数据加载到模型中
model.fit(dfuf)

# 输出模型估计结果
params = model.inspect()
print("模型参数估计结果：")
print(params)
params.to_excel(os.path.join(output_path,"SEM模型参数.xlsx"), index=False)

# 计算拟合度
fit = calc_stats(model) 
print("\n拟合度指标：")
print(fit.T)
fit.to_excel(os.path.join(output_path,"SEM拟合度.xlsx"), index=False)


''' --------------------------------------- '''
''' 结构方程作图 '''

# 变量字典
varname_mapping = {
    'User': '用户信息',
    'Age': '年龄',
    'Spending': '月消费额',
    'Gender_Female': '性别（女性=1）',
    'Identity_graduate': '身份（研究生）',
    'Identity_part': '身份（兼职）',
    'Identity_work': '身份（工作）',
    'Emo': '情绪状态',
    'Loneliness': '孤独感',
    'Depre': '抑郁度',
    'Reason': '使用AI的原因',
    'ReasonLonely': '缓解孤独',
    'ReasonEmo': '缓解情绪',
    'ReasonSupport': '获取支持',
    'ReasonSocial': '社交互动',
    'ReasonEnter': '娱乐放松',
    'ReasonNovel': '新奇探索',
    'ReasonOther': '其他原因',
    'Exper': 'AI使用体验',
    'ExperOverall': '总体体验',
    'ExperNature': '自然交互',
    'ExperSupport': '支持性',
    'ExperPrivacy': '隐私性',
    'ExperPersonal': '个性化',
    'Impor': 'AI功能偏好',
    'ImporNature': '偏好-自然交互',
    'ImporSupport': '偏好-支持性',
    'ImporPrivacy': '偏好-隐私性',
    'ImporPersonal': '偏好-个性化',
    'Scenario': '使用场景',
    'ScenarioSleep': '场景-助眠',
    'ScenarioWork': '场景-工作',
    'ScenarioDown': '场景-情绪低落',
    'ScenarioNovel': '场景-探索',
    'ScenarioTravel': '场景-出行',
    'ScenarioOther': '场景-其他',
    'NotReason': '不使用AI的原因',
    'NotReasonProduct': '不使用-功能不足',
    'NotReasonNece': '不使用-不必要',
    'NotReasonPrivacy': '不使用-隐私担忧',
    'NotReasonOther': '不使用-其他',
    'Accept': 'AI接受度',
    'Try': '尝试意愿',
    'ConcernSocial': '担忧-社交风险',
    'ConcernMoral': '担忧-伦理风险',
    'ConcernPrivacy': '担忧-隐私风险',
    'ConcernExper': '担忧-体验风险',
    'ConcernNot': '担忧-不使用理由',
    'ConcernOther': '担忧-其他',
    'WTP': 'AI支付意愿',
    'Pay': '付费意愿',
    'Price': '付费价格',
    'Market': 'AI市场预期',
}

# 固定测量模型结构
measurement_model = {
    '用户信息': ['年龄', '月消费额', '性别（女性=1）', '身份（研究生）', '身份（兼职）', '身份（工作）'],
    '情绪状态': ['孤独感', '抑郁度'],
    'AI接受度': ['尝试意愿', '担忧-社交风险', '担忧-伦理风险', '担忧-隐私风险', '担忧-体验风险', '担忧-不使用理由', '担忧-其他'],
    'AI支付意愿': ['付费意愿', '付费价格'],
}

# 提取结构模型路径关系
params = params[params['p-value'] != '-'].copy()
params['p-value'] = pd.to_numeric(params['p-value'], errors='raise')

# 筛选结构模型路径关系 (op == '~')
structural_params = params[params['op'] == '~']

# 构建结构模型字典 {因变量: {自变量: (Estimate, p)}}
structural_model = dict()
for _, row in structural_params.iterrows():
    dep = row['lval']
    indep = row['rval']
    estimate = row['Estimate']
    p_value = row['p-value']  # 现在100% float

    dep_cn = varname_mapping.get(dep, dep)
    indep_cn = varname_mapping.get(indep, indep)

    if dep_cn not in structural_model:
        structural_model[dep_cn] = dict()
    structural_model[dep_cn][indep_cn] = (estimate, p_value)


# 作图
dot = Digraph(comment='SEM Path Diagram')

# 配色（更浅、更柔和）
latent_color = '#fbd5a1'        # 浅杏色
observed_color = '#fff3b0'      # 奶油黄
structural_edge_color = '#a0d9cb'  # 浅蓝绿
measurement_edge_color = '#a3bfc8'  # 浅灰蓝

# 全局布局调整
dot.attr(dpi='400')
dot.attr('graph', rankdir='LR', ranksep='1.5', nodesep='1.5', size='10,16!', ratio='compress')

# 字体和节点样式
dot.attr('node', fontname='SimSun', fontsize='100', width='5.6', height='3.2', margin='0.4,0.3')
dot.attr('edge', fontname='SimSun', fontsize='80')

# 添加测量模型
for latent, observables in measurement_model.items():
    dot.node(latent, latent, style='filled', fillcolor=latent_color, shape='ellipse', fontcolor='black')
    for obs in observables:
        dot.node(obs, obs, style='filled', fillcolor=observed_color, shape='box', fontcolor='black')
        dot.edge(latent, obs, color=measurement_edge_color, penwidth='12')

# 添加结构模型
for dependent, independents in structural_model.items():
    dot.node(dependent, dependent, style='filled', fillcolor=latent_color, shape='ellipse', fontcolor='black')
    for independent, (estimate, p_value) in independents.items():
        edge_style = 'solid' if p_value < 0.05 else 'dashed'
        label = f'{estimate:.2f}'
        dot.edge(independent, dependent, color=structural_edge_color, penwidth='12', style=edge_style, label=label, fontsize='80')

# 输出图像
dot.render(filename = 'Vertical_SEM_Path_Diagram', directory = output_path, format='png', cleanup=True)



    
    
    
    
    




    
    