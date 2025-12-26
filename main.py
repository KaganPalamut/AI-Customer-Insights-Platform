import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer # TF-IDF yerine bazen Count daha stabildir
from sklearn.naive_bayes import MultinomialNB

print("📂 Model eğitiliyor...")
df = pd.read_csv("veriler_dev.csv")

# 1. Vektörleştirme
vectorizer = CountVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(df['text'].values.astype('U'))
y = df['label']

# 2. Model (En güvenli liman: Naive Bayes)
model = MultinomialNB(alpha=1.0)
model.fit(X, y)

# 3. Kaydet
with open("sentiment_model.pkl", "wb") as f: pickle.dump(model, f)
with open("tfidf_vectorizer.pkl", "wb") as f: pickle.dump(vectorizer, f)

print("🎉 Model 'Başarılı' kelimesini hafızasına kazıdı! Test edebilirsin.")