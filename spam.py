import sklearn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction import stop_words
from sklearn.linear_model import logistic
from sklearn import svm

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
def spam_or_not(s):
    if s == 'ham':
        return 0
    elif s == 'spam':
        return 1

df = pd.read_csv('sms-spam.csv', encoding='latin1', usecols=[0, 1], names=['is spam', 'sms'], header=None)[1:]
df['is spam'] = df['is spam'].apply(spam_or_not)

#df.head(20)
#df.info()

#df.groupby('is spam').describe()

'''
plt.title('SMS Category')
labels = ['spam', 'ham']
sizes = [4825, 747]
colors = ['r', 'g']
plt.pie(x=sizes, labels=labels, colors=colors)
plt.plot()
'''

def p_digits(text):
    count = 0
    for i in text:
        if i.isdigit():
            count += 1
    return (count/ len(text))*100


df['p_digits'] = df['sms'].apply(p_digits)
#def.head()

'''
plt.hist(df[df['is spam'] == 0]['p_digits'], range=(0, 20), bins=10, rwidth=1)
plt.xlabel('p_digits')
plt.ylabel('number of sms')
plt.title('non spam percentage')
plt.show()
'''

def p_ques_mark(text):
    count = 0
    for i in text:
        if i == '?':
            count +=1
    return (count/len(text)) * 100

df['p_ques_mark']=df['sms'].apply(p_ques_mark)


'''
plt.hist(df[df['is spam'] == 0]['p_ques_mark'], range=(0, 4), bins=10, rwidth=1)
plt.xlabel('p_ques_mark')
plt.ylabel('number of sms')
plt.title('non spam percentage')
plt.show()
'''

def p_excl_mark(text):
    count = 0
    for i in text:
        if i == '!':
            count += 1
    return (count / len(text)) * 100


df['p_excl_mark'] = df['sms'].apply(p_excl_mark)


'''
plt.hist(df[df['is spam'] == 0]['p_excl_mark'], range=(0, 4), bins=10, rwidth=1)
plt.xlabel('p_excl_mark')
plt.ylabel('number of sms')
plt.title('non spam percentage')
plt.show()
'''

def p_caps(text):
    count = 0
    for i in text:
        if i.isupper():
            count += 1
    return (count / len(text)) * 100


df['p_caps'] = df['sms'].apply(p_caps)

'''
plt.hist(df[df['is spam'] == 0]['p_caps'], range=(0, 40), bins=10, rwidth=1)
plt.xlabel('p caps')
plt.ylabel('number of sms')
plt.title('non spam percentage')
plt.show()
'''
df['length_of_sms']=df['sms'].apply(len)


def thanks(text):
  counter = 0
  for i in text:
    if i == 'win' or i=='award' or i=='free' or i=='thankyou' or i=='Won' or i=='credit' or i=='Win' or i=="Free" or i=='Thankyou':
      counter +=1
  return counter

df['thanks']=df['sms'].apply(thanks)



def isalpha(word):
    word = word.replace('.', '')
    return word.isalpha()

def clean_sms(text):
    text = text.lower()
    return (' '.join(filter(lambda s: isalpha(s) and s not in stopwords.words("english"), text.split()))).replace('.','').split()



#nltk.download('stopwords')
import nltk
nltk.download('stopwords')
cv = CountVectorizer(strip_accents='ascii', min_df=5, analyzer=clean_sms)
df = pd.concat([df, pd.DataFrame(cv.fit_transform(df['sms']).todense(), columns=cv.get_feature_names(), index=np.arange(1, cv.fit_transform(df['sms']).todense().shape[0] + 1))], axis=1)

X = df.drop(['is spam', 'sms'], axis=1)
y = df['is spam']
#print(X, y)

X_train, X_other, y_train, y_other = train_test_split(X, y, test_size=0.4, random_state=42)
X_cv, X_test, y_cv, y_test = train_test_split(X_other, y_other, test_size=0.5, random_state=42)

clf = LogisticRegression()
clf = clf.fit(X_train, y_train)

y_actual = y_test
y_predicted = clf.predict(X_test)

true_positives = X_test[(y_actual == 1) & (y_predicted == 1)]
true_negatives = X_test[(y_actual == 0) & (y_predicted == 0)]
false_positives = X_test[(y_actual == 0) & (y_predicted == 1)]
false_negatives = X_test[(y_actual == 1) & (y_predicted == 0)]

precision = true_positives.shape[0] / (true_positives.shape[0] + false_positives.shape[0])
print("Precision:", precision)
recall = true_positives.shape[0] / (true_positives.shape[0] + false_negatives.shape[0])
print("Recall:", recall)
f1_score = 2 * precision * recall / (precision + recall)
print("F1 score:", f1_score)