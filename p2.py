import pandas as pd
import numpy as np
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

df = pd.read_csv("./aleksandre_khokhiashvili_2_86214954.csv")

X = df[["words", "links", "capital_words", "spam_word_count"]]
y = df["is_spam"]
# split data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=0x1337
)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

print(f'Resulting model coefficients: {model.coef_}')

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:')
print(conf_matrix)


spam_words = open('./spam-words-EN.txt').read().splitlines()
# make all spam words lowercase
spam_words = [word.lower() for word in spam_words]

# "words", "links", "capital_words", "spam_word_count"
def data_from_email(email_text: str):
    url_extract_pattern = "https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)"
    words = len(email_text.split())
    links = len(re.findall(url_extract_pattern, email_text))
    capital_words = 0
    spam_word_count = 0
    for spam_word in spam_words:
        if spam_word in email_text.lower():
            spam_word_count += 1
    for word in email_text.split():
        # check first letter of the word
        if word[0] in string.ascii_uppercase:
            capital_words += 1
    return np.array([words, links, capital_words, spam_word_count])

def check_is_spam(email: str):
    X = data_from_email(email)
    return model.predict(np.array([X]))[0]


good_email = open('email_good.txt').read()
bad_email = open('email_bad.txt').read()

# read good and bad email
X_good = data_from_email(good_email)
X_bad = data_from_email(bad_email)

# print the good and bad inputs
print(f'X_good = {X_good}')
print(f'X_bad = {X_bad}')

print('good email is_spam = ', check_is_spam(good_email))
print('bad email is_spam = ', check_is_spam(bad_email))




