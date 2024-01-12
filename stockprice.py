import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from textblob import TextBlob

def analyze_sentiment(tweet):
    analysis = TextBlob(str(tweet))
    return analysis.polarity

# The rest of your code...


def analyze_sentiment(tweet):
    analysis = TextBlob(str(tweet))
    return analysis.polarity

# Load data
news = pd.read_csv("Combined_News_DJIA.csv", low_memory=False, parse_dates=[0])

# Splitting the dataset into training and testing
train_news = news[news['Date'] < '2014-07-15']
test_news = news[news['Date'] >= '2014-07-15']

# Feature extraction using CountVectorizer
vectorizer = CountVectorizer(min_df=0.01, max_df=0.8)
train_news_list = [' '.join(str(k) for k in row[2:27]) for _, row in train_news.iterrows()]
test_news_list = [' '.join(str(x) for x in row[2:27]) for _, row in test_news.iterrows()]

news_vector = vectorizer.fit_transform(train_news_list)
test_vector = vectorizer.transform(test_news_list)

# Logistic Regression model
lr = LogisticRegression()
model = lr.fit(news_vector, train_news["Label"])
predictions = model.predict(test_vector)

# Evaluate the model
accuracy1 = accuracy_score(test_news["Label"], predictions)
print("Baseline Model Accuracy:", accuracy1)

# Continue with other models and analysis...

# Example: Bi-gram
nvectorizer = TfidfVectorizer(min_df=0.05, max_df=0.85, ngram_range=(2, 2))
news_nvector = nvectorizer.fit_transform(train_news_list)

# Continue with the rest of your code...

# Example: Random Forest - Bi-gram
nvectorizer_rf = TfidfVectorizer(min_df=0.01, max_df=0.95, ngram_range=(2, 2))
news_nvector_rf = nvectorizer_rf.fit_transform(train_news_list)
rfmodel = RandomForestClassifier(random_state=55)
rfmodel.fit(news_nvector_rf, train_news["Label"])

# Continue with the rest of your code...

# Example: Naive Bayes
nvectorizer_nb = TfidfVectorizer(min_df=0.05, max_df=0.8, ngram_range=(2, 2))
news_nvector_nb = nvectorizer_nb.fit_transform(train_news_list)
nbmodel = MultinomialNB(alpha=0.5)
nbmodel.fit(news_nvector_nb, train_news["Label"])

# Continue with the rest of your code...

# Example: Gradient Boosting
nvectorizer_gb = TfidfVectorizer(min_df=0.05, max_df=0.8, ngram_range=(2, 2))
news_nvector_gb = nvectorizer_gb.fit_transform(train_news_list)
gbmodel = GradientBoostingClassifier(random_state=52)
gbmodel.fit(news_nvector_gb, train_news["Label"])

# Continue with the rest of your code...

# Example: Trigram
n3vectorizer = TfidfVectorizer(min_df=0.0004, max_df=0.115, ngram_range=(3, 3))
news_n3vector = n3vectorizer.fit_transform(train_news_list)

# Continue with the rest of your code...

# Sentiment Analysis
train_sentiment = train_news.drop(['Date', 'Label'], axis=1)
for column in train_sentiment:
    train_sentiment[column] = train_sentiment[column].apply(analyze_sentiment)
train_sentiment = train_sentiment + 10

# Continue with the rest of your code...

# XGBoost Model
XGB_model = XGBClassifier()
gradiant = XGB_model.fit(train_sentiment, train_news['Label'])
test_sentiment = test_news.drop(['Date', 'Label'], axis=1)
for column in test_sentiment:
    test_sentiment[column] = test_sentiment[column].apply(analyze_sentiment)
test_sentiment = test_sentiment + 10
y_pred = gradiant.predict(test_sentiment)

# Evaluate the sentiment model
conf_matrix = confusion_matrix(test_news['Label'], y_pred)
print("Confusion Matrix of the Gradient Boosting Sentiment Model:\n", conf_matrix)

sentiment_accuracy = accuracy_score(test_news['Label'], y_pred)
print("Sentiment Accuracy:", sentiment_accuracy)

f1 = f1_score(test_news['Label'], y_pred, average='weighted')
print("F1 Score:", f1)