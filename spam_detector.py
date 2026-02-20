# Spam Email Detection using Machine Learning

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    "text": [
        "Win a free iPhone now",
        "Limited time offer claim your prize",
        "Meeting scheduled at 10 am",
        "Project deadline tomorrow",
        "Congratulations you won lottery",
        "Let's have lunch today"
    ],
    "label": [1,1,0,0,1,0]
}

df = pd.DataFrame(data)

# Split data
X = df["text"]
y = df["label"]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train,y_train)

# Test accuracy
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test,y_pred))

# Test custom message
msg = ["Free offer just for you"]
msg = vectorizer.transform(msg)
prediction = model.predict(msg)

if prediction[0] == 1:
    print("Spam message")
else:
    print("Not spam")
