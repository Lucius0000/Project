# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 01:04:49 2025

@author: Lucius
"""

import os
import pandas as pd
import jieba
import pickle
import gc
import numpy as np
from tqdm import tqdm
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# ========== 参数 ==========
index_path = "整理后公司报告/定期报告-最终索引表.xlsx"
text_root = "整理后公司报告"
stopword_path = "可用数据/cn_stopwords.txt"
output_dir = "结果"
os.makedirs(output_dir, exist_ok=True)
checkpoint_path = os.path.join(output_dir, "checkpoint_tfidf.pkl")
batch_size = 2500
max_features = 20000
svd_dim = 20

# 加载停用词
def load_stopwords(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())
stopwords = load_stopwords(stopword_path)

# 加载索引
df_index = pd.read_excel(index_path, dtype=str)
df_index = df_index.dropna(subset=["文件名", "股票代码", "发布日期"])

def get_file_path_by_stock_code(filename, root_dir):
    stock_code = filename.split("_")[0]
    for folder in os.listdir(root_dir):
        if folder.startswith(stock_code + "_"):
            return os.path.join(root_dir, folder, filename)
    return None

df_index["文本路径"] = df_index["文件名"].apply(lambda x: get_file_path_by_stock_code(x, text_root))
df_index = df_index.dropna(subset=["文本路径"])

# 文本生成器
def yield_documents(df):
    for _, row in df.iterrows():
        try:
            with open(row["文本路径"], "r", encoding="utf-8") as f:
                content = f.read()
            if len(content.strip()) < 100:
                continue
            words = jieba.lcut(content)
            words = [w for w in words if len(w) > 1 and w not in stopwords]
            yield " ".join(words), row
        except:
            continue

# 初始化
start_index = 0
batch_id = 0
vectorizer = None

if os.path.exists(checkpoint_path):
    print("检测到断点，正在恢复...")
    with open(checkpoint_path, "rb") as f:
        ckpt = pickle.load(f)
        start_index = ckpt["start_index"]
        batch_id = ckpt["batch_id"]
        vectorizer = ckpt["vectorizer"]

print("正在分批提取 TF-IDF 特征...")
texts, metas = [], []
saved_batches = []
total_docs = len(df_index)

for i, (text, row) in enumerate(tqdm(yield_documents(df_index), total=total_docs)):
    if i < start_index:
        continue
    texts.append(text)
    metas.append(row)

    if len(texts) >= batch_size:
        try:
            if vectorizer is None:
                print(f"正在拟合 TF-IDF 第 {batch_id+1} 批...")
                vectorizer = TfidfVectorizer(max_features=max_features)
                tfidf_batch = vectorizer.fit_transform(texts)
            else:
                tfidf_batch = vectorizer.transform(texts)

            sparse.save_npz(os.path.join(output_dir, f"tfidf_batch_{batch_id}.npz"), tfidf_batch)
            pd.DataFrame(metas).to_pickle(os.path.join(output_dir, f"meta_batch_{batch_id}.pkl"))
            saved_batches.append(batch_id)

            with open(checkpoint_path, "wb") as f:
                pickle.dump({
                    "start_index": i + 1,
                    "batch_id": batch_id + 1,
                    "vectorizer": vectorizer
                }, f)

            print(f"已保存第 {batch_id+1} 批 TF-IDF")

            texts, metas = [], []
            gc.collect()
            batch_id += 1
        except MemoryError:
            print(f"内存错误，第 {batch_id+1} 批失败，释放内存重试...")
            gc.collect()
            continue

# 处理剩余文本
if texts:
    try:
        tfidf_batch = vectorizer.transform(texts)
        sparse.save_npz(os.path.join(output_dir, f"tfidf_batch_{batch_id}.npz"), tfidf_batch)
        pd.DataFrame(metas).to_pickle(os.path.join(output_dir, f"meta_batch_{batch_id}.pkl"))
        saved_batches.append(batch_id)
        print(f"已保存第 {batch_id+1} 批 TF-IDF（最后一批）")
    except MemoryError:
        print("内存错误，最后一批跳过")
    texts, metas = [], []
    gc.collect()

# 合并并降维
print("正在拼接所有 TF-IDF 特征并降维...")
tfidf_all = []
meta_all = []
for bid in range(batch_id):
    tfidf_all.append(sparse.load_npz(os.path.join(output_dir, f"tfidf_batch_{bid}.npz")))
    meta_all.append(pd.read_pickle(os.path.join(output_dir, f"meta_batch_{bid}.pkl")))

X = sparse.vstack(tfidf_all)
df_meta = pd.concat(meta_all, ignore_index=True)
del tfidf_all, meta_all

svd = TruncatedSVD(n_components=svd_dim, random_state=42)
X_svd = svd.fit_transform(X)

df_svd = pd.DataFrame(X_svd, columns=[f"SVD特征{i+1}" for i in range(svd_dim)])
df_result = pd.concat([df_meta.reset_index(drop=True), df_svd], axis=1)
df_result.to_csv(os.path.join(output_dir, "TFIDF_SVD_特征向量.csv"), index=False, encoding="utf-8-sig")
print("所有结果保存完毕：TFIDF_SVD_特征向量.csv")

# 导出每个 SVD 特征的关键词（按权重排序）
print("正在导出每个 SVD 特征的关键词...")
terms = vectorizer.get_feature_names_out()
topn = 10  # 每个维度输出前10个关键词

svd_keywords = []
for i, comp in enumerate(svd.components_):
    top_indices = comp.argsort()[::-1][:topn]
    svd_keywords.append([terms[j] for j in top_indices])

df_svd_keywords = pd.DataFrame(svd_keywords)
df_svd_keywords.columns = [f"关键词{j+1}" for j in range(topn)]
df_svd_keywords.index = [f"SVD特征{i+1}" for i in range(svd.n_components)]

df_svd_keywords.to_csv(os.path.join(output_dir, "SVD_关键词列表.csv"), encoding="utf-8-sig")
print("SVD 特征关键词已保存完毕！")



