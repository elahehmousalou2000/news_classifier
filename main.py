from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

texts = [
    "new football team wins match",
    "government announces new policy",
    "new AI model released",
    "player scores two goals",
]
labels = ["sports", "politics", "technology", "sports"]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = LogisticRegression()
model.fit(X, labels)

news = input("News: ")
print(model.predict(vectorizer.transform([news]))[0])
