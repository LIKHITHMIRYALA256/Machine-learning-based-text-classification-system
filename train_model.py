import pandas as pd
import pickle
from preprocessing import clean_text

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

print("🔹 Loading dataset...")

df = pd.read_csv("data/spam.csv", encoding="latin-1")

print("🔹 Dataset loaded")
print("Columns:", df.columns)

# Keep required columns
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

print("🔹 Columns renamed")

# Clean text
df['clean_text'] = df['text'].apply(clean_text)

X = df['clean_text']
y = df['label']

print("🔹 Text cleaned")

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

print("🔹 Vectorization done")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

print("🔹 Data split")

# Models
nb_model = MultinomialNB()
lr_model = LogisticRegression(max_iter=1000)

nb_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)

print("🔹 Models trained")

# Evaluation
nb_pred = nb_model.predict(X_test)
lr_pred = lr_model.predict(X_test)

print("\nNaive Bayes Accuracy:", accuracy_score(y_test, nb_pred))
print(classification_report(y_test, nb_pred))

print("\nLogistic Regression Accuracy:", accuracy_score(y_test, lr_pred))
print(classification_report(y_test, lr_pred))

# SAVE MODELS
pickle.dump(lr_model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model and vectorizer saved successfully!")
