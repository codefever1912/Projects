import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import nltk
import re
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("punkt_tab")

data = pd.read_csv("./Datasets/twitter_data.csv")
data = data.dropna()
data['category'] = data['category'].astype(int)

def preprocess(tweet):
    #Regex handling and tokenization
    tweet = tweet.lower()
    tweet = re.sub(r'[^a-zA-Z\s]', '', tweet) #removes non-alphabet characters
    tweet = re.sub(r'http\S+|www|S+|https\S+', '', tweet, flags=re.MULTILINE) #removes links and URLs
    tweet = re.sub(r'@\w+', '', tweet) #removes @ mentions
    tweet = re.sub(r'#', '', tweet)#removes hashtags

    tokenized_tweet = ''.join(word_tokenize(tweet)) #Tokenizes the tweet, separating tweet into individual tokens and joining them to form a single long string of characters

    #Encoding based on ascii values
    preprocessed_tweet = []
    for ch in tokenized_tweet:
        preprocessed_tweet.append(ord(ch)) #Converts the sinlge long string into a sequence of numbers corresponding to the ASCII values of the characters

    return preprocessed_tweet

tweets = data['clean_text']
preprocessed_tweets = [preprocess(tweet) for tweet in tweets]

#Padding the tweets
final_tweets = []
padding_len = max(len(tweet) for tweet in preprocessed_tweets)
for tweet in preprocessed_tweets:
    final_tweets.append(tweet + [0] * (padding_len - len(tweet)))

X = np.array(final_tweets)
y = np.array(data['category'])

scaler = StandardScaler()
scaled_data = scaler.fit_transform(X,y)

x_train, x_test, y_train, y_test = train_test_split(scaled_data,y,test_size=0.2,random_state=42)

analyzer = DecisionTreeClassifier()
analyzer.fit(x_train, y_train)

predictions = analyzer.predict(x_test)
print(accuracy_score(y_test, predictions))