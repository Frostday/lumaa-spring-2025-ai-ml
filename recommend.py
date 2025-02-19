import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import nltk
import sys
import warnings

warnings.filterwarnings("ignore")

nltk.download("stopwords", quiet=True)
english_stopwords = set(nltk.corpus.stopwords.words('english'))

# TF-IDF vectorizer initialized
tfidf_vectorizer = TfidfVectorizer(
    tokenizer = nltk.word_tokenize,
    stop_words = list(english_stopwords),
    ngram_range = (1,2),
    max_df = 1.0,
    min_df = 10
)

# Get names and features of movies
df = pd.read_csv("data/movie_dataset.csv")
df["genres"].fillna("", inplace=True)
df["keywords"].fillna("", inplace=True)
df["tagline"].fillna("", inplace=True)
df['overview'] = df['genres'] + ' ' + df['keywords'] + ' ' + df['overview'] + ' ' + df['tagline']
df = df[["original_title", "overview"]]
df.dropna(inplace=True)
names = df["original_title"].values
features = tfidf_vectorizer.fit_transform(df["overview"])

# Get recommendations
def get_recommendations(text, top=5):
    X = tfidf_vectorizer.transform([text])
    similarities = cosine_similarity(X, features)[0]
    indices = np.argsort(similarities)[-top:][::-1]
    print("Recommended Movies:")
    recs = [f"{names[i]} ({similarities[i]:.2f})" for i in indices]
    print(' | '.join(recs))

if len(sys.argv)==2:
    get_recommendations(sys.argv[1])
elif len(sys.argv)==3:
    get_recommendations(sys.argv[1], top=int(sys.argv[2]))
else:
    print("Extra arguments given")
