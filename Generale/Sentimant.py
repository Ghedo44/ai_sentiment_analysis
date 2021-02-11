# Import tweepy to work with the twitter API
import tweepy as tw

# Import numpy and pandas to work with dataframes
import numpy as np
import pandas as pd

# Import seaborn and matplotlib for viz
from matplotlib import pyplot as plt

consumer_key = 'BeaPnmFUSAkmbUcbg78AyWLvz'
consumer_secret = 'sVOPpN9vS7HAfRHjtWG99ojkn0X9hP0jIpCuTn1pQLaGVGvrr0'
access_token = '865844470079770624-bztOyHL7evKB7uFN9vsyrcf5t4yLu3t'
access_token_secret = 'GgdnLfDhk8qOJaNBpI2KSI7J5wyDYm2J8EvlzwbaEL2WV'

# Authenticate
auth = tw.OAuthHandler(consumer_key, consumer_secret)
# Set Tokens
auth.set_access_token(access_token, access_token_secret)
# Instantiate API
api = tw.API(auth, wait_on_rate_limit=True)

hashtag = "#GME"
query = tw.Cursor(api.search, q=hashtag).items(1000)
tweets = [{'Tweet':tweet.text, 'Timestamp':tweet.created_at} for tweet in query]
print(tweets)

df = pd.DataFrame.from_dict(tweets)
df.head()

gme_handle = ['gme', 'GME', 'Gamestop', 'gamestop', 'GME\'s']

def identify_subject(tweet, refs):
    flag = 0
    for ref in refs:
        if tweet.find(ref) != -1:
            flag = 1
    return flag

df['GME'] = df['Tweet'].apply(lambda x: identify_subject(x, gme_handle))

# Import stopwords
import nltk
from nltk.corpus import stopwords

# Import textblob
from textblob import Word, TextBlob

nltk.download('stopwords')
nltk.download('wordnet')
stop_words = stopwords.words('english')
custom_stopwords = ['RT', '#PresidentialDebate']

def preprocess_tweets(tweet, custom_stopwords):
    processed_tweet = tweet
    processed_tweet.replace('[^\w\s]', '')
    processed_tweet = " ".join(word for word in processed_tweet.split() if word not in stop_words)
    processed_tweet = " ".join(word for word in processed_tweet.split() if word not in custom_stopwords)
    processed_tweet = " ".join(Word(word).lemmatize() for word in processed_tweet.split())
    return(processed_tweet)

df['Processed Tweet'] = df['Tweet'].apply(lambda x: preprocess_tweets(x, custom_stopwords))
df.head()

print('Base review\n', df['Tweet'][0])
print('\n------------------------------------\n')
print('Cleaned and lemmatized review\n', df['Processed Tweet'][0])

# Calculate polarity
df['polarity'] = df['Processed Tweet'].apply(lambda x: TextBlob(x).sentiment[0])
df['subjectivity'] = df['Processed Tweet'].apply(lambda x: TextBlob(x).sentiment[1])

#display(df[df['Trump']==1][['Trump','polarity','subjectivity']].groupby('Trump').agg([np.mean, np.max, np.min, np.median]))
df[df['GME']==1][['GME','polarity','subjectivity']].groupby('GME').agg([np.mean, np.max, np.min, np.median])

gme = df[df['GME']==1][['Timestamp', 'polarity']]
gme = gme.sort_values(by='Timestamp', ascending=True)
gme['MA Polarity'] = gme.polarity.rolling(10, min_periods=3).mean()


repub = 'red'
demo = 'blue'
fig, axes = plt.subplots(1, 1, figsize=(13, 10))

axes.plot(gme['Timestamp'], gme['MA Polarity'])
axes.set_title("\n".join(["GME Polarity"]))

fig.suptitle("\n".join(["Presidential Debate Analysis"]), y=0.98)

plt.show()
