#!pip install tweepy
import tweepy
import pandas as pd


# In[4]: python -m spacy download en_core_web_lg
# https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.0.0/en_core_web_lg-3.0.0-py3-none-any.whl
#pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-2.2.0/en_core_web_lg-2.2.0.tar.gz

# Authentication
consumer_key = 'BeaPnmFUSAkmbUcbg78AyWLvz'
consumer_secret = 'sVOPpN9vS7HAfRHjtWG99ojkn0X9hP0jIpCuTn1pQLaGVGvrr0'
access_token = '865844470079770624-bztOyHL7evKB7uFN9vsyrcf5t4yLu3t'
access_token_secret = 'GgdnLfDhk8qOJaNBpI2KSI7J5wyDYm2J8EvlzwbaEL2WV'
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)


# In[49]:


number_of_tweets = 200
tweets = []
likes = []
time = []

#api.search, q="Dogecoin"
try:
    for i in tweepy.Cursor(api.user_timeline, id="elonmusk", tweet_mode="extended").items(number_of_tweets):
        tweets.append(i.full_text)
        likes.append(i.favorite_count)
        time.append(i.created_at)
except:
    print('Errore di connessione')


# In[54]:


df = pd.DataFrame({'tweets':tweets, 'likes':likes, 'time':time})
df = df[~df.tweets.str.contains("RT")]
df = df.reset_index(drop=True)

print(df)
# In[56]:


mostlike = df.loc[df.likes.nlargest(5).index]
print(mostlike)


# In[ ]:


#!pip install spacy
import matplotlib.pyplot as plt
import re
import spacy
nlp = spacy.load('en_core_web_lg')
import seaborn as sns


# In[ ]:


list_of_sentences = [sentence for sentence in df.tweets]

lines = []
for sentence in list_of_sentences:
    words = sentence.split()
    for w in words:
        lines.append(w)


# In[ ]:


lines = [re.sub(r'[^A-Za-z0-9]+', '', x) for x in lines]

lines2 = []

for word in lines:
    if word != '':
        lines2.append(word)


# In[ ]:



#This is stemming the words to their root
from nltk.stem.snowball import SnowballStemmer

# The Snowball Stemmer requires that you pass a language parameter
s_stemmer = SnowballStemmer(language='english')

stem = []
for word in lines2:
    stem.append(s_stemmer.stem(word))
    
print(stem)


# In[ ]:


#Removing all Stop Words

stem2 = []

for word in stem:
    if word not in nlp.Defaults.stop_words:
        stem2.append(word)

print(stem2)


# In[ ]:


df = pd.DataFrame(stem2)

df = df[0].value_counts()

#df
#df['freq'] = df.groupby(0)[0].transform('count')
#df['freq'] = df.groupby(0)[0].transform('count')
#df.sort_values(by = ('freq'), ascending=False)


# In[ ]:


#This will give frequencies of our words

from nltk.probability import FreqDist

freqdoctor = FreqDist()

for words in df:
    freqdoctor[words] += 1

print(freqdoctor)


# In[ ]:


#This is a simple plot that shows the top 20 words being used
#df.plot(20)

df = df[:20,]
plt.figure(figsize=(10,5))
sns.barplot(df.values, df.index, alpha=0.8)
plt.title('Top Words Overall')
plt.ylabel('Word from Tweet', fontsize=12)
plt.xlabel('Count of Words', fontsize=12)
plt.show()


# In[ ]:


import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
# get_ipython().system('python -m spacy download en_core_web_sm')
nlp = en_core_web_sm.load()


# In[ ]:


def show_ents(doc):
    if doc.ents:
        for ent in doc.ents:
            print(ent.text + ' - ' + ent.label_ + ' - ' + str(spacy.explain(ent.label_)))


# In[ ]:


str1 = " " 
stem2 = str1.join(lines2)

stem2 = nlp(stem2)

label = [(X.text, X.label_) for X in stem2.ents]

df6 = pd.DataFrame(label, columns=['Word', 'Entity'])

df7 = df6.where(df6['Entity'] == 'ORG')

df7 = df7['Word'].value_counts()


# In[ ]:


df = df7[:20, ]
plt.figure(figsize=(10, 5))
sns.barplot(df.values, df.index, alpha=0.8)
plt.title('Top Organizations Mentioned')
plt.ylabel('Word from Tweet', fontsize=12)
plt.xlabel('Count of Words', fontsize=12)
plt.show()


# In[ ]:


str1 = " " 
stem2 = str1.join(lines2)

stem2 = nlp(stem2)

label = [(X.text, X.label_) for X in stem2.ents]

df10 = pd.DataFrame(label, columns=['Word', 'Entity'])

df10 = df10.where(df10['Entity'] == 'PERSON')

df11 = df10['Word'].value_counts()


# In[ ]:


df = df11[:20, ]
plt.figure(figsize=(10, 5))
sns.barplot(df.values, df.index, alpha=0.8)
plt.title('Top People Mentioned')
plt.ylabel('Word from Tweet', fontsize=12)
plt.xlabel('Count of Words', fontsize=12)
plt.show()
