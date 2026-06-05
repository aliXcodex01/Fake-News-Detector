import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

# ---------------------------
# LOAD DATASET
# ---------------------------

fake_path = "data/Fake.csv"
real_path = "data/True.csv"

fake_df = pd.read_csv(fake_path)
real_df = pd.read_csv(real_path)

fake_df["label"] = 0
real_df["label"] = 1

df = pd.concat([fake_df, real_df])

df = df[["text", "label"]]

print("Total Samples:", len(df))

# ---------------------------
# SPLIT DATA
# ---------------------------

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------
# TF-IDF
# ---------------------------

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---------------------------
# MODEL
# ---------------------------

model = PassiveAggressiveClassifier(
    max_iter=1000
)

model.fit(X_train_vec, y_train)

# ---------------------------
# EVALUATION
# ---------------------------

predictions = model.predict(X_test_vec)

accuracy = accuracy_score(
    y_test,
    predictions
)

print(f"Accuracy: {accuracy*100:.2f}%")

# ---------------------------
# SAVE MODEL
# ---------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/fake_news_model.pkl"
)

joblib.dump(
    vectorizer,
    "models/vectorizer.pkl"
)

print("Model Saved Successfully")