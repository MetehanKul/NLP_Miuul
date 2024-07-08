from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from wordcloud import WordCloud
import nltk


filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)


# 1. Text Preprocessing


df = pd.read_csv("wiki_data.csv" , index_col=0)
df.head()

    
def clean_data(x):
    # Normalizing Case Folding
    x = x.str.lower()
    # Punctuations
    x = x.str.replace(r'[^\w\s]', '', regex=True)
    x = x.str.replace(r'\n', '', regex=True)
    # Numbers
    x = x.str.replace(r'\d', ' ', regex=True)
    return x


df['text'] = clean_data(df['text'])
df.head()

## StopWords

sw = stopwords.words('english')

df["text"] = df["text"].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))
df["text"].head()

## Rarewords

sil = pd.Series(" ".join(df["text"]).split()).value_counts()
drops = sil[sil <= 1]
drops.head()

df["text"] = df["text"].apply(lambda x: " ".join(x for x in x.split() if x not in drops))
df['text'].head()

# Tokenization

## nltk.download("punkt")

df["text"].apply(lambda x: TextBlob(x).words).head()


# Lemmatization

df['text'] = df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

# 2. Text Visualization

tf = df["text"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()

tf.columns = ["words", "tf"]

tf.sort_values("tf", ascending=False)

## BarPlot


tf[tf["tf"] > 2500].plot.bar(x="words", y="tf")
plt.show()

## WordCloud

text = " ".join(i for i in df.text)


wordcloud = WordCloud(max_font_size=50,
                      max_words=100,
                      background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()



