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

# İgnore the code

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)


# Read the data

df_copy = pd.read_csv("spotify_reviews.csv")
df_copy.head()

df = df_copy[["content" , "score"]][:7500]
df.head()

def clean_data(x):
    # Normalizing Case Folding
    x = x.str.lower()
    # Punctuations
    x = x.str.replace(r'[^\w\s]', ' ', regex=True)
    x = x.str.replace(r'\n', ' ', regex=True)
    # Numbers
    x = x.str.replace(r'\d', ' ', regex=True)
    return x

df['content'] = clean_data(df['content'])
df.head()

# Stopwords

sw = stopwords.words("english")
df["content"] = df["content"].apply(lambda x : " ".join(x for x in str(x).split() if x not in sw ))
df.content.head()

# Rarewords

temp_df = pd.Series(" ".join(df["content"]).split()).value_counts()

drops = temp_df[temp_df <= 1]

df["content"] = df["content"].apply(lambda x : " ".join(x for x in str(x).split() if x not in drops ))
df.content.head()

# Leamtization 


df["content"] = df["content"].apply(lambda x : " ".join([Word(word).lemmatize() for word in x.split()]))


# Text Visualization

tf = df["content"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()

tf.columns = ["words", "tf"]

tf.sort_values("tf", ascending=False)

tf[tf["tf"] > 1000].plot.bar(x="words", y="tf")
plt.show()

# Wordcloud

content = " ".join(i for i in df.content)

wordcloud = WordCloud(max_font_size =55,
                      max_words = 125,
                      background_color= "white"
                      ).generate(content)

plt.figure()
plt.imshow(wordcloud , interpolation = "bilinear")
plt.axis("off")
plt.show()

# Sentiment Analysis

sia = SentimentIntensityAnalyzer()
sia.polarity_scores("this film is so good")


df["polarity_scores"] = df["content"].apply(lambda x: sia.polarity_scores(x)["compound"])
df.head()

# Feature Engineering

df["sentiment"] = df["content"].apply(lambda x : "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")
df.sample(10)

df.groupby("sentiment")["score"].mean() # Pos score and Neg score 

df["sentiment"] = LabelEncoder().fit_transform(df["sentiment"])

df["sentiment"].sample(10)

x = df["content"]
y = df["sentiment"]

# Count Vectorizer
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X_count = vectorizer.fit_transform(x)

vectorizer.get_feature_names()[10:15]
X_count.toarray()[10:15]

# TF-IDF Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

tf_idf_word_vectorizer = TfidfVectorizer()
X_tf_idf_word = tf_idf_word_vectorizer.fit_transform(x)

tf_idf_ngram_vectorizer = TfidfVectorizer(ngram_range=(2, 3))
X_tf_idf_ngram = tf_idf_ngram_vectorizer.fit_transform(x)

## Model Time (Logistic Regression)

log_model = LogisticRegression().fit(X_count , y)

cross_val_score(log_model ,
                X_count , 
                y ,
                cv = 5,
                scoring = "accuracy").mean() ## Model Score = 0.86


log_model = LogisticRegression().fit(X_tf_idf_word , y)

cross_val_score(log_model ,
                X_tf_idf_word , 
                y ,
                cv = 5,
                scoring = "accuracy").mean() # Model Scoring = 0.83


log_model = LogisticRegression().fit(X_tf_idf_ngram , y)

cross_val_score(log_model ,
                X_tf_idf_ngram , 
                y ,
                cv = 5,
                scoring = "accuracy").mean() # Model Scoring = 0.69


## Model Time (Random Forest)

rf_model = RandomForestClassifier().fit(X_count , y)
cross_val_score(rf_model , X_count , y ,cv=5 , n_jobs=-1).mean()  ## Model Scoring = 0.77

rf_model = RandomForestClassifier().fit(X_tf_idf_word , y)
cross_val_score(rf_model , X_tf_idf_word , y ,cv=5 , n_jobs=-1).mean()  ## Model Scoring = 0.77

rf_model = RandomForestClassifier().fit(X_tf_idf_ngram , y)
cross_val_score(rf_model , X_tf_idf_ngram , y ,cv=5 , n_jobs=-1).mean()  ## Model Scoring = 0.67

# Model Tuned

rf_model = RandomForestClassifier()

rf_params = {"max_depth" : [8,None],
             "max_features" : [7,"auto"],
             "min_samples_split" : [2,5],
             "n_estimators" : [50 , 100 ]}

rf_grids = GridSearchCV(rf_model ,
                        rf_params ,
                        cv = 5 ,
                        n_jobs = -1,
                        verbose = 1).fit(X_count , y)

rf_grids.best_params_

rf_final = rf_model.set_params(**rf_grids.best_params_).fit(X_count , y)

cross_val_score(rf_final , X_count , y , cv = 5 , n_jobs = -1).mean() ## Random Forest Final Models Score = 72.5 

## XGB

from xgboost import XGBClassifier

xgb_model = XGBClassifier().fit(X_count , y)

cross_val_score(xgb_model , X_count , y , cv = 5 , n_jobs = -1).mean() ## XGB Model Score = 0.84 

# Done