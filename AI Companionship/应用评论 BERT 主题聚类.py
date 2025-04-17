import pandas as pd
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel(r"C:\Users\Lucius\Desktop\NLP\应用 bert\星野_bert.xlsx")
df.dropna(subset=['评论'], inplace=True)

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
embeddings = model.encode(df['评论'].tolist())

topic_model = BERTopic(language='chinese (simplified)')
topics, probs = topic_model.fit_transform(df['评论'].tolist(), embeddings)

df['topic'] = topics

df['sentiment'] = df['sentiment_score'].apply(lambda x: "正面" if x > 0.7 else "负面" if x < 0.4 else "中性")

topic_counts = df['topic'].value_counts().sort_index() 
sentiment_score_avg = df.groupby('topic')['sentiment_score'].mean() 
sentiment_avg = df.groupby('topic')['sentiment'].apply(lambda x: x.mode()[0]) 


topic_info = topic_model.get_topic_info()
topic_info.set_index('Topic', inplace=True)
topic_info['comment_count'] = topic_counts
topic_info['sentiment_score_avg'] = sentiment_score_avg
topic_info['sentiment_avg'] = sentiment_avg
topic_info.reset_index(inplace=True)

plt.figure(figsize=(12, 5))
sns.barplot(data=topic_info, x="Topic", y="sentiment_score_avg", palette="coolwarm")
plt.xlabel("Topic ID")
plt.ylabel("平均 Sentiment Score")
plt.title("星野：各主题的情感得分均值")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 5))
sns.barplot(data=topic_info, x="Topic", y="comment_count", palette="viridis")
plt.xlabel("Topic ID")
plt.ylabel("评论数量")
plt.title("星野：各主题的评论数量")
plt.xticks(rotation=45)
plt.show()

sentiment_topic_dist = df.groupby(['sentiment', 'topic']).size().unstack(fill_value=0)

plt.figure(figsize=(12, 6))
sns.heatmap(sentiment_topic_dist, cmap="Blues", annot=True, fmt="d")
plt.xlabel("Topic ID")
plt.ylabel("Sentiment")
plt.title("星野：不同情感类别下的主题分布")
plt.show()

output_file_1 = r"C:\Users\Lucius\Desktop\星野_主题分析结果_评论.xlsx"
df.to_excel(output_file_1, index=False)  
print(f"结果已保存到 {output_file_1}")

output_file_2 = r"C:\Users\Lucius\Desktop\星野_主题分析结果_情感.xlsx"
with pd.ExcelWriter(output_file_2) as writer:
    topic_info.to_excel(writer, sheet_name="Topic Info")
    sentiment_topic_dist.to_excel(writer, sheet_name="Sentiment Topic Distribution")

print(f"结果已保存到 {output_file_2}")

import matplotlib.pyplot as plt
import seaborn as sns

sentiment_topic_dist = df.groupby(['topic', 'sentiment']).size().unstack(fill_value=0)
sentiment_topic_dist['total'] = sentiment_topic_dist.sum(axis=1)
sentiment_ratio = sentiment_topic_dist.div(sentiment_topic_dist['total'], axis=0) * 100
sentiment_ratio.drop(columns=['total'], inplace=True)
sentiment_ratio = sentiment_ratio[['负面', '中性', '正面']]
colors = ['#FF7F7F', '#D3D3D3', '#4682B4'] 

plt.figure(figsize=(12, 6))
sentiment_ratio.plot(kind='bar', stacked=True, color=colors, alpha=0.85)

plt.xlabel("主题编号 (Topic ID)")
plt.ylabel("评论占比 (%)")
plt.title("星野 - 主题情感分布")
plt.legend(title="情感类别", labels=['负面', '中性', '正面'], loc='center left', bbox_to_anchor=(1, 0.5)) 

plt.xticks(rotation=45)
plt.ylim(0, 100) 
plt.grid(axis='y', linestyle="--", alpha=0.5)

plt.show()

from wordcloud import WordCloud
import numpy as np

topic_keywords = topic_model.get_topics()

word_freq = {}
for topic_id, words in topic_keywords.items():
    if topic_id == -1: 
        continue
    weight = topic_info.loc[topic_info['Topic'] == topic_id, 'comment_count'].values[0]
    for word, score in words:
        word_freq[word] = word_freq.get(word, 0) + score * weight

wordcloud = WordCloud(
    font_path=r"C:\Windows\Fonts\simhei.ttf", 
    background_color="white",
    width=800,
    height=400,
    max_words=200,
    colormap="coolwarm",
).generate_from_frequencies(word_freq)
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off") 
plt.title("星野 - 主题关键词词云", fontsize=14)
plt.show()


