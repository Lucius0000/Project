# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 16:36:50 2025

@author: Lucius
"""


import pandas as pd

df = pd.read_excel(r"C:\Users\Lucius\Desktop\Cos Love虚拟情感聊天-虚拟恋人的最新评论和评分-应用宝官网.xlsx")

#去除特殊字符 
import re

def clean_text(text):
    text = re.sub(r"[^\u4e00-\u9fa5]", " ", text)  # 只保留中文
    text = re.sub(r"\s+", " ", text).strip()  # 去掉多余空格
    return text

df["评论"] = df["评论"].apply(clean_text)

#去重
df.drop_duplicates(subset=["用户", "评论"], inplace=True)

df = df[["评论","点赞数","评论数"]]


from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 加载预训练 BERT 模型
tokenizer = BertTokenizer.from_pretrained("uer/roberta-base-finetuned-jd-binary-chinese")
model = BertForSequenceClassification.from_pretrained("uer/roberta-base-finetuned-jd-binary-chinese")

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).tolist()[0]
    
    return probs[1]  # 取正面情感的概率

df["sentiment_score"] = df["评论"].apply(predict_sentiment)
df["sentiment"] = df["sentiment_score"].apply(lambda x: "正面" if x > 0.7 else "负面" if x < 0.4 else "中性")

df.to_excel(r"C:\Users\Lucius\Desktop\Cos Love_bert.xlsx" ,index = False)


