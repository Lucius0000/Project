# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 16:36:50 2025

@author: Lucius
"""


import pandas as pd

df = pd.read_stata(r"小红书 merge.dta")

import re

def clean_text(text):
    text = re.sub(r"[^\u4e00-\u9fa5]", " ", text)  # 只保留中文
    text = re.sub(r"\s+", " ", text).strip() 
    return text

df["帖子标题"] = df["帖子标题"].apply(clean_text)
df["帖子正文"] = df["帖子正文"].apply(clean_text)
df["评论内容"] = df["评论内容"].apply(clean_text)


from transformers import BertTokenizer, BertForSequenceClassification
import torch
tokenizer = BertTokenizer.from_pretrained("uer/roberta-base-finetuned-jd-binary-chinese")
model = BertForSequenceClassification.from_pretrained("uer/roberta-base-finetuned-jd-binary-chinese")

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).tolist()[0]
    
    return probs[1] 

df["帖子标题_sentiment_score"] = df["帖子标题"].apply(predict_sentiment)
df["帖子标题_sentiment"] = df["帖子标题_sentiment_score"].apply(lambda x: "正面" if x > 0.7 else "负面" if x < 0.4 else "中性")
df["帖子正文_sentiment_score"] = df["帖子正文"].apply(predict_sentiment)
df["帖子正文_sentiment"] = df["帖子正文_sentiment_score"].apply(lambda x: "正面" if x > 0.7 else "负面" if x < 0.4 else "中性")

def predict_comment_sentiment(comment):
    if pd.isna(comment) or comment == "":
        return None, None
    sentiment_score = predict_sentiment(comment)
    sentiment = "正面" if sentiment_score > 0.7 else "负面" if sentiment_score < 0.4 else "中性"
    return sentiment_score, sentiment
df["评论内容_sentiment_score"], df["评论内容_sentiment"] = zip(*df["评论内容"].apply(predict_comment_sentiment))


df.to_excel(r"C:\Users\Lucius\Desktop\小红书_bert.xlsx" ,index = False)


