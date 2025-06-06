# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 14:31:22 2025

@author: Lucius
"""


import os
import re
import gc
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, r2_score

# ----- 参数与路径 -----
TRAIN_FOLDER = r'C:\Users\Lucius\Downloads\2025泰迪杯 赛题\B\示例数据\附件1'
TEST_FOLDER  = r'C:\Users\Lucius\Downloads\2025泰迪杯 赛题\B\示例数据\附件2'
RESULT_FOLDER = r'C:\Users\Lucius\Downloads\2025泰迪杯 赛题\B\B题-全部数据\result_sedentary_classifier'
os.makedirs(RESULT_FOLDER, exist_ok=True)

WINDOW_SIZE_SEC = 25 * 60      # 25分钟
TOLERANCE_RATIO = 0.2          # 20% 非久坐容忍
KEYWORDS = ['sitting', 'lying', 'desk', 'office']

# ----- 日志 -----
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# ----- 标签函数 -----
def extract_met_value(met_str):
    m = re.search(r'MET\s*([\d\.]+)', str(met_str))
    return float(m.group(1)) if m else None

def is_sedentary(met, label):
    if pd.isna(met) or pd.isna(label):
        return 0
    return int((met < 1.6) and any(k in label.lower() for k in KEYWORDS))

# ----- 加载元数据 -----
meta1 = pd.read_csv(os.path.join(TRAIN_FOLDER, 'Metadata1.csv'))
meta2 = pd.read_csv(os.path.join(TEST_FOLDER, 'Metadata2.csv'))
age_map = {'18-29':0,'30-37':1,'38-52':2,'53+':3}
meta1['age']=meta1['age'].map(age_map)
meta2['age']=meta2['age'].map(age_map)

# ----- 加载并构建训练集 -----
log('加载训练数据...')
dtypes = {'x':'float32','y':'float32','z':'float32','time':'str'}
train_list=[]
for f in os.listdir(TRAIN_FOLDER):
    if not (f.startswith('P') and f.endswith('.csv')): continue
    df=pd.read_csv(os.path.join(TRAIN_FOLDER,f),dtype=dtypes)
    pid=f[:-4]
    if 'annotation' not in df: continue
    df=df.dropna(subset=['annotation'])
    df[['label','MET']]=df['annotation'].str.rsplit(';',n=1,expand=True)
    df['MET']=df['MET'].apply(extract_met_value)
    df=df.dropna(subset=['MET'])
    df['sedentary']=df.apply(lambda r: is_sedentary(r['MET'],r['label']),axis=1)
    m=meta1[meta1['pid']==pid].iloc[0]
    df['age']=m['age']; df['sex']=1 if m['sex']=='M' else 0
    df['time']=pd.to_datetime(df['time'],errors='coerce')
    df['qidx']=df['time'].dt.hour*4+df['time'].dt.minute//15
    df['dow']=df['time'].dt.dayofweek
    df['acc_mag']=np.sqrt(df['x']**2+df['y']**2+df['z']**2)
    train_list.append(df[['x','y','z','acc_mag','age','sex','qidx','dow','sedentary']])
train_df=pd.concat(train_list,ignore_index=True)
X=train_df.drop(columns='sedentary')
y=train_df['sedentary']
del train_list, train_df; gc.collect()

# 划分验证集
X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)

# ----- GPU 检测 -----
log('检测 GPU 支持...')
params={'objective':'binary','metric':'binary_logloss','boosting_type':'gbdt',
        'learning_rate':0.05,'verbose':-1,'max_bin':255,'device':'gpu',
        'gpu_platform_id':0,'gpu_device_id':0}
try:
    _=lgb.train(params,lgb.Dataset(X_train,label=y_train),num_boost_round=1)
    use_gpu=True; log('GPU 可用，启用 GPU 加速')
except:
    use_gpu=False; log('GPU 不可用，使用 CPU 训练')
    params.pop('device',None)

# ----- 训练模型 -----
log('训练模型...')
train_ds=lgb.Dataset(X_train,label=y_train)
model=lgb.train(params,train_ds,num_boost_round=100)

# ----- 性能评估 -----
y_val_pred=model.predict(X_val)
print(classification_report(y_val,(y_val_pred>0.5).astype(int)))
r2=r2_score(y_val,y_val_pred)
log(f"R² score: {r2:.4f}")

# ----- 特征重要性 -----
log('绘制特征重要性图...')
plt.figure(figsize=(8,6))
lgb.plot_importance(model,max_num_features=15,importance_type='gain')
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig(os.path.join(RESULT_FOLDER,'feature_importance.png'),dpi=300)
plt.close()

# ----- 测试集预测 & 区间识别 -----
log('预测测试集并识别区间...')
segments_list=[]
summary_list=[]

dtypes.update({'time':'str'})
for f in os.listdir(TEST_FOLDER):
    if not (f.startswith('P') and f.endswith('.csv')): continue
    df=pd.read_csv(os.path.join(TEST_FOLDER,f),dtype=dtypes)
    pid=f[:-4]
    m=meta2[meta2['pid']==pid].iloc[0]
    df['age']=m['age']; df['sex']=1 if m['sex']=='M' else 0
    df['time']=pd.to_datetime(df['time'],errors='coerce')
    df['qidx']=df['time'].dt.hour*4+df['time'].dt.minute//15
    df['dow']=df['time'].dt.dayofweek
    df['acc_mag']=np.sqrt(df['x']**2+df['y']**2+df['z']**2)

    feats=df[['x','y','z','acc_mag','age','sex','qidx','dow']]
    df['sed_pred']=model.predict(feats)>0.5
    df.to_csv(os.path.join(RESULT_FOLDER,f),index=False)

    # 区间检测
    arr=df['sed_pred'].values; times=df['time']
    min_pts=WINDOW_SIZE_SEC*100
    i=0
    bouts=[]
    while i<len(df):
        j=i; total=0; sedc=0
        while j<len(df) and (times.iloc[j]-times.iloc[i]).total_seconds()<=3600:
            total+=1; sedc+=arr[j]
            if total>=min_pts and sedc/total>=1-TOLERANCE_RATIO:
                bouts.append((times.iloc[i],times.iloc[j]))
                i=j+1; break
            j+=1
        i+=1
    # 记录
    for s,e in bouts:
        dur=(e-s).total_seconds()/60
        segments_list.append({'pid':pid,'start_time':s,'end_time':e,'duration_min':dur})
    total_min=sum([(e-s).total_seconds() for s,e in bouts])/60
    summary_list.append({'pid':pid,'total_sed_min':round(total_min,1)})
    log(f"{pid} 检测出 {len(bouts)} 区间，总时长 {total_min:.1f} 分钟")

# 保存表格
seg_df=pd.DataFrame(segments_list)
seg_df.to_csv(os.path.join(RESULT_FOLDER,'sedentary_intervals.csv'),index=False)
sum_df=pd.DataFrame(summary_list)
sum_df.to_csv(os.path.join(RESULT_FOLDER,'sedentary_summary.csv'),index=False)

# ----- 久坐区间可视化（条带状） -----
log('绘制区间条带图...')
plt.figure(figsize=(12, 6))
y_labels=sorted(seg_df['pid'].unique())
for idx,pid in enumerate(y_labels):
    grp=seg_df[seg_df['pid']==pid]
    for _,row in grp.iterrows():
        plt.hlines(idx, row['start_time'], row['end_time'], linewidth=6)
plt.yticks(range(len(y_labels)), y_labels)
plt.xlabel('Time'); plt.ylabel('PID')
plt.title('Sedentary Intervals per Participant')
plt.tight_layout()
plt.savefig(os.path.join(RESULT_FOLDER,'intervals_strip.png'),dpi=300)
plt.close()

# ----- 时段分布图（柱状） -----
log('绘制时段分布图...')
seg_df['hour']=seg_df['start_time'].dt.hour
hc=seg_df['hour'].value_counts().sort_index()
plt.figure(figsize=(10,4))
hc.plot(kind='bar')
plt.xlabel('Hour of Day'); plt.ylabel('Bout Count')
plt.title('Sedentary Bout Distribution by Hour')
plt.tight_layout()
plt.savefig(os.path.join(RESULT_FOLDER,'bout_hourly_dist.png'),dpi=300)
plt.close()

log('全部完成。')
