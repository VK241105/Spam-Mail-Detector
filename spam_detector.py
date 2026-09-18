import pandas as pd
import urllib.request
import zipfile

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt


# -----------------------------------------
# 1. Download the dataset
# -----------------------------------------

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_file = "sms_spam_collection.zip"

print("Downloading dataset...")

urllib.request.urlretrieve(url, zip_file)

with zipfile.ZipFile(zip_file, "r") as zip_ref:
    zip_ref.extractall(".")

print("Dataset downloaded successfully!")


# -----------------------------------------
# 2. Load the dataset
# -----------------------------------------

data = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

print("\nDataset shape:", data.shape)

print("\nClass distribution:")
print(data["label"].value_counts())


# -----------------------------------------
# 3. Convert labels into numbers
# -----------------------------------------

data["label"] = data["label"].map({
    "ham": 0,
    "spam": 1
})


# -----------------------------------------
# 4. Split data into training and testing
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    data["message"],
    data["label"],
    test_size=0.2,
    random_state=42,
    stratify=data["label"]
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# -----------------------------------------
# 5. Convert text into TF-IDF features
# -----------------------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF features created!")


# -----------------------------------------
# 6. Train Naive Bayes model
# -----------------------------------------

model = MultinomialNB()

model.fit(X_train_tfidf, y_train)

print("Naive Bayes model trained!")


# -----------------------------------------
# 7. Make predictions
# -----------------------------------------

y_pred = model.predict(X_test_tfidf)


# -----------------------------------------
# 8. Evaluate the model
# -----------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n-----------------------------")
print("MODEL PERFORMANCE")
print("-----------------------------")

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Ham", "Spam"]
    )
)


# -----------------------------------------
# 9. Confusion Matrix
# -----------------------------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Ham", "Spam"],
    yticklabels=["Ham", "Spam"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Spam Mail Detector - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()


# -----------------------------------------
# 10. Test your own messages
# -----------------------------------------

print("\n-----------------------------")
print("TEST YOUR OWN MESSAGE")
print("-----------------------------")

while True:

    message = input(
        "\nEnter a message (or type 'exit' to stop): "
    )

    if message.lower() == "exit":
        print("Program ended.")
        break

    message_tfidf = vectorizer.transform([message])

    prediction = model.predict(message_tfidf)[0]

    if prediction == 1:
        print("🚨 Prediction: SPAM")
    else:
        print("✅ Prediction: HAM (Not Spam)")