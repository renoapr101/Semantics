# -*- coding: utf-8 -*-
"""Livin_inweek.ipynb
import pandas as pd

file_path = '/content/livin_reviews.xlsx'
df = pd.read_excel(file_path)

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

display(df)

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

all_text = ' '.join(df['content'].astype(str))
vectorizer = CountVectorizer()
word_counts = vectorizer.fit_transform([all_text])
word_list = vectorizer.get_feature_names_out()
count_list = word_counts.toarray().flatten()
word_index_df = pd.DataFrame({'word': word_list, 'count': count_list})
word_index_df = word_index_df.sort_values(by='count', ascending=False).reset_index(drop=True)
display(word_index_df)
word_index_df.to_csv('word_counts_index.csv', index=False)
print("Word index with counts exported to 'word_counts_index.csv'")

df = df[['content', 'score']]
df = df.dropna()
df[['content', 'score']]

"""#Sentimen"""

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

df['sentiment'] = df['score'].apply(lambda x: 'positif' if x >= 4 else ('negatif' if x <= 2 else 'netral'))
display(df[['content', 'score', 'sentiment']])

df['score'].value_counts()

import seaborn as sns
result = df.groupby(['score']).size()
sns.barplot(x = result.index, y = result.values)

for sentiment_label, sentiment_df in df.groupby('sentiment'):
  print(f"Reviews with sentiment: {sentiment_label}")
  display(sentiment_df[['content']])

import matplotlib.pyplot as plt

sentiment_counts = df['sentiment'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Sentiment')
plt.axis('equal')
plt.show()

# Commented out IPython magic to ensure Python compatibility.
df_p=df[df['sentiment']=='positif']
all_words_lem = ' '.join([word for word in df_p['content']])
# %matplotlib inline
import matplotlib.pyplot as plt
from wordcloud import WordCloud
wordcloud = WordCloud(background_color='white', width=800, height=500, random_state=21, max_font_size=130).generate(all_words_lem)
plt.figure(figsize=(20, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off');

# Commented out IPython magic to ensure Python compatibility.
df_p=df[df['sentiment']=='negatif']
all_words_lem = ' '.join([word for word in df_p['content']])
# %matplotlib inline
import matplotlib.pyplot as plt
from wordcloud import WordCloud
wordcloud = WordCloud(background_color='white', width=800, height=500, random_state=21, max_font_size=130).generate(all_words_lem)
plt.figure(figsize=(20, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off');

# Commented out IPython magic to ensure Python compatibility.
df_p=df[df['sentiment']=='netral']
all_words_lem = ' '.join([word for word in df_p['content']])
# %matplotlib inline
import matplotlib.pyplot as plt
from wordcloud import WordCloud
wordcloud = WordCloud(background_color='white', width=800, height=500, random_state=21, max_font_size=130).generate(all_words_lem)
plt.figure(figsize=(20, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off');

"""#Groupin"""

import re
keywords = [
    'anggaran', 'pengeluaran', 'budget', 'pembukuan', 'kas', 'jadual', 'jadwal', 'terjadual', 'terjadwal', 'autodebet', 'kantong',
    'bayar kredit', 'bulanan', 'pemasukan', 'finansial', 'alokasi', 'budgeting', 'spending', 'spent', 'save', 'saving',
    'dompet', 'spending', 'tagihan', 'hemat', 'terjadual',
    'catatan keuangan', 'pengingat', 'streak', 'nudges', 'laporan keuangan'
]
pattern = r'\b(?:' + '|'.join(re.escape(k) for k in keywords) + r')\b'
mask = df['content'].apply(lambda x: isinstance(x, str) and bool(re.search(pattern, x.lower())))
financial_reviews = df[mask]
print("Reviews containing financial keywords:")
display(financial_reviews[['content', 'score', 'sentiment']])

nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian'))

new_stopwords = {'yg', 'nya', 'ga', 'livin', 'livin mandiri', 'mandiri', 'dan', 'admin bulanan', 'kartu kredit'}
stop_words.update(new_stopwords)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

financial_reviews['cleaned_content'] = financial_reviews['content'].apply(clean_text)

print("\nCleaned financial reviews:")
display(financial_reviews[['content', 'cleaned_content', 'score', 'sentiment']])

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
all_words_financial = ' '.join([str(review) for review in financial_reviews['cleaned_content']])

# %matplotlib inline
wordcloud_financial = WordCloud(background_color='white', width=800, height=500, random_state=21, max_font_size=130).generate(all_words_financial)
plt.figure(figsize=(20, 10))
plt.imshow(wordcloud_financial, interpolation='bilinear')
plt.axis('off');
plt.title('Word Cloud of Financial Reviews')
plt.show()
