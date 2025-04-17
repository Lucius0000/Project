# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 18:36:39 2025

@author: Lucius
"""

import math
import numpy as np
import pandas as pd
from gensim import corpora, models
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import random

df = pd.read_excel(r"C:\Users\Lucius\Desktop\NLP\小红书_bert.xlsx")

def convert_to_number(value):
    if isinstance(value, str):
        if '万' in value:
            return float(value.replace('万', '')) * 10000  # 转换 "2.1万" -> 21000
        elif value in ['点赞', '收藏', '评论']:
            return 0 
    try:
        return float(value)  
        return None  
df[['点赞数', '收藏数', '评论数']] = df[['点赞数', '收藏数', '评论数']].applymap(convert_to_number)
df = df.fillna(0).astype({'点赞数': 'int', '收藏数': 'int', '评论数': 'int'})

# 1. 按帖子聚合
group_keys = ['标题', '正文', '点赞数', '评论数', '标题情感评分', '正文情感评分']
def agg_comments(x):
    valid = x.dropna(subset=['评论内容'])
    if valid.empty:
        return pd.Series({'评论内容_agg': '', '评论情感评分_agg': np.nan})
    else:
        content = " ".join(valid['评论内容'].astype(str))
        score = valid['评论情感评分'].astype(float).mean()
        return pd.Series({'评论内容_agg': content, '评论情感评分_agg': score})
df_grouped = df.groupby(group_keys).apply(agg_comments).reset_index()

# 2. 定义权重函数
def weight_func(count):
    return math.log(1 + count)

scale = 10 

# 3. 构造加权文档
def construct_weighted_doc(row):
    wt_like = weight_func(row['点赞数'])
    rep_title = ((row['标题'] + " ") * int(wt_like * scale)) if wt_like > 0 else row['标题']
    rep_content = ((row['正文'] + " ") * int(wt_like * scale)) if wt_like > 0 else row['正文']
    if pd.notna(row['评论内容_agg']) and row['评论数'] > 0:
        wt_comment = weight_func(row['评论数']) * 0.5
        rep_comment = ((row['评论内容_agg'] + " ") * int(wt_comment * scale)) if wt_comment > 0 else row['评论内容_agg']
    else:
        rep_comment = ""
    return rep_title + rep_content + rep_comment

df_grouped['加权文档'] = df_grouped.apply(construct_weighted_doc, axis=1)

# 4. 构造情感得分
def compute_post_sentiment(row):
    wt_title = weight_func(row['点赞数'])
    wt_content = weight_func(row['点赞数'])
    total_wt = wt_title + wt_content
    sentiment_sum = row['标题情感评分'] * wt_title + row['正文情感评分'] * wt_content
    if pd.notna(row['评论情感评分_agg']) and row['评论数'] > 0:
        wt_comment = weight_func(row['评论数']) * 0.5
        sentiment_sum += row['评论情感评分_agg'] * wt_comment
        total_wt += wt_comment
    return sentiment_sum / total_wt if total_wt != 0 else 0

df_grouped['帖子情感评分'] = df_grouped.apply(compute_post_sentiment, axis=1)

# 5. 主题建模
vectorizer = CountVectorizer(stop_words='english')
doc_term_matrix = vectorizer.fit_transform(df_grouped['加权文档'])
dictionary = corpora.Dictionary([vectorizer.get_feature_names_out()])
corpus = [dictionary.doc2bow(doc.split()) for doc in df_grouped['加权文档']]
lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=10)
topics = lda_model.show_topics(num_topics=5, num_words=10, formatted=False)
topic_keywords = {topic_id: [word for word, prob in words] for topic_id, words in topics}

def get_dominant_topic(doc_bow):
    topic_probs = lda_model.get_document_topics(doc_bow)
    if topic_probs:
        return max(topic_probs, key=lambda x: x[1])[0]
    else:
        return None

df_grouped['主题'] = [get_dominant_topic(doc) for doc in corpus]

# 6. 计算主题情感倾向（加权平均）
def total_weight(row):
    wt = weight_func(row['点赞数']) * 2
    if pd.notna(row['评论情感评分_agg']) and row['评论数'] > 0:
        wt += weight_func(row['评论数']) * 0.5
    return wt

df_grouped['总权重'] = df_grouped.apply(total_weight, axis=1)
topic_sentiments = df_grouped.groupby('主题').apply(
    lambda g: np.average(g['帖子情感评分'], weights=g['总权重'])
).reset_index(name='主题情感评分')

# 7. 构造导出结果表格
result_table = topic_sentiments.copy()
result_table['关键词'] = result_table['主题'].apply(lambda t: ", ".join(topic_keywords.get(t, [])))
def classify_sentiment(score):
    if score > 0.7:
        return "正面"
    elif score < 0.4:
        return "负面"
    else:
        return "中性"
result_table['情感分类'] = result_table['主题情感评分'].apply(classify_sentiment)

def summarize_topic(t):
    kws = topic_keywords.get(t, [])
    if kws:
        return "该主题主要讨论：" + "，".join(kws)
    else:
        return "暂无信息"
result_table['主题含义'] = result_table['主题'].apply(summarize_topic)

def get_representative_text(topic_id, top_n=2):
    sub_df = df_grouped[df_grouped['主题'] == topic_id]
    # 按总权重降序排序
    sub_df = sub_df.sort_values(by='总权重', ascending=False)
    # 拼接标题和正文，选取前 top_n 条
    texts = sub_df.apply(lambda row: row['标题'] + " " + row['正文'], axis=1).head(top_n).tolist()
    return " | ".join(texts)

result_table['代表文本'] = result_table['主题'].apply(lambda t: get_representative_text(t, top_n=2))

result_table.rename(columns={'主题': '主题编号'}, inplace=True)
output_path = r"C:\Users\Lucius\Desktop\NLP\主题建模结果.xlsx"
result_table.to_excel(output_path, index=False)
print("导出结果表格路径：", output_path)
print(result_table)

# 8. 可视化
topic_weight_sum = df_grouped.groupby('主题')['总权重'].sum().reset_index()
topic_weight_sum['主题'] = topic_weight_sum['主题'].astype(str)

plt.figure(figsize=(8, 4))
color_list = ['#f8d8a4', '#ebb089', '#588797', '#a8c3d9', '#f2b880', '#d8e3f1']
colors = [color_list[i % len(color_list)] for i in range(len(topic_weight_sum))]
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.bar(topic_weight_sum['主题'], topic_weight_sum['总权重'], color=colors)
plt.xlabel('主题编号')
plt.ylabel('帖子数量（加权）')
plt.title('小红书各主题下的帖子数量（加权）')
plt.show()

# 9. 词云图
all_words = {}
for idx, row in result_table.iterrows():
    t = row['主题编号']
    for word in topic_keywords.get(t, []):
        all_words[word] = row['主题情感评分']

def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    palette = ["#f8d8a4", "#ebb089", "#588797", "#a8c3d9", "#f2b880", "#d8e3f1"]
    return random.choice(palette)

wc = WordCloud(font_path='C:/Windows/Fonts/simhei.ttf', width=800, height=400, background_color='white')
wc.generate_from_frequencies(all_words)

plt.figure(figsize=(10, 5))
plt.imshow(wc.recolor(color_func=color_func, random_state=3), interpolation="bilinear")
plt.axis('off')
plt.title("小红书主题关键词词云图")
plt.show()

topic_weight_sum = df_grouped.groupby('主题')['总权重'].sum().reset_index()
topic_weight_sum['主题'] = topic_weight_sum['主题'].astype(str)

plt.figure(figsize=(8, 4))
color_list = ['#f8d8a4', '#ebb089', '#588797', '#a8c3d9', '#f2b880', '#d8e3f1']
colors = [color_list[i % len(color_list)] for i in range(len(result_table))]
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.bar(result_table['主题编号'].astype(str), result_table['主题情感评分'], color=colors)
plt.xlabel('主题编号')
plt.ylabel('情感得分')
plt.title('小红书各主题情感倾向得分')
plt.show()
