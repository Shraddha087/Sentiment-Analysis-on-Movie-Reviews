import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# -------------------------------
# 1. Load data
# -------------------------------
df = pd.read_csv("data/movie_reviews.csv")

# -------------------------------
# 2. Preprocessing Function
# -------------------------------
lemmatizer = WordNetLemmatizer()
# Ensure required NLTK resources are available
for pkg, path in [('punkt', 'tokenizers/punkt'), ('punkt_tab', 'tokenizers/punkt_tab/english'), ('stopwords', 'corpora/stopwords'), ('wordnet', 'corpora/wordnet')]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(pkg)

stop_words = stopwords.words("english")

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words]
    return " ".join(tokens)

df["clean_review"] = df["review"].apply(preprocess)

# If dataset is very small, train on entire dataset without grid search
n_samples = len(df)
if n_samples < 4:
    print(f"Small dataset detected ({n_samples} samples). Training on the full dataset without grid search.")
    X = df["clean_review"]
    y = df["sentiment"]
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(max_iter=200))
    ])
    pipeline.fit(X, y)
    os.makedirs("model", exist_ok=True)
    joblib.dump(pipeline, "model/sentiment_model.pkl")
    print("Model saved to model/sentiment_model.pkl (trained on full small dataset)")
    raise SystemExit(0)

# -------------------------------
# 3. Train-test split
# -------------------------------
X = df["clean_review"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# 4. Pipeline (TF-IDF + Logistic Regression)
# -------------------------------
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=200))
])

# -------------------------------
# 5. Hyperparameter tuning
# -------------------------------
params = {
    "tfidf__ngram_range": [(1,1), (1,2)],
    "clf__C": [0.1, 1, 3, 10]
}

grid = GridSearchCV(pipeline, params, cv=3, scoring='accuracy', n_jobs=-1)
# Handle very small training sets (avoid cv > n_samples)
n_train = len(X_train)
if n_train < 2:
    raise ValueError(f"Not enough training samples ({n_train}). Need at least 2 samples to train.")

cv_folds = min(3, n_train)
if cv_folds < 2:
    # fallback: fit pipeline without grid search
    print("Very small dataset — fitting pipeline without cross-validated grid search.")
    pipeline.fit(X_train, y_train)
    grid = None
else:
    grid = GridSearchCV(pipeline, params, cv=cv_folds, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)

# -------------------------------
# 6. Evaluation
# -------------------------------
if grid is None:
    y_pred = pipeline.predict(X_test)
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
else:
    y_pred = grid.predict(X_test)
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# 7. Save the model
# -------------------------------
os.makedirs("model", exist_ok=True)
if grid is None:
    joblib.dump(pipeline, "model/sentiment_model.pkl")
    print("Model saved to model/sentiment_model.pkl (pipeline)")
else:
    joblib.dump(grid.best_estimator_, "model/sentiment_model.pkl")
    print("Model saved to model/sentiment_model.pkl")
