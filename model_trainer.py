import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from trnlp import TrnlpWord
import pickle

nltk.download("stopwords")

df = pd.read_csv("turkce_yorumlar_duygu_1000.csv")

stop_words = set(stopwords.words("turkish"))
lemmatizer = TrnlpWord()

def clear_and_lemmatize(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r"[^\w\s]", "", sentence)
    sentence = re.sub(r"\d+", "", sentence)
    all_words = sentence.split()
    clear = []
    for word in all_words:
        if word not in stop_words:
            lemmatizer.setword(word)
            lemma = lemmatizer.get_stem
            clear.append(lemma if lemma else word)
    return " ".join(clear)

df["clean_comment"] = df["yorum"].apply(clear_and_lemmatize)

X = df["clean_comment"]
y = df["duygu"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

vector = TfidfVectorizer()
X_train_tfidf = vector.fit_transform(X_train)
X_test_tfidf = vector.transform(X_test)

model = XGBClassifier(eval_metric="mlogloss") 
model.fit(X_train_tfidf, y_train)

with open("xgboost_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vector, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)


"""
# 8. TEST VE RAPOR
y_pred = model.predict(X_test_tfidf)

y_test_labels = le.inverse_transform(y_test)
y_pred_labels = le.inverse_transform(y_pred)

print("XGBoost Model Sonuçları:")
print(classification_report(y_test_labels, y_pred_labels))"""

