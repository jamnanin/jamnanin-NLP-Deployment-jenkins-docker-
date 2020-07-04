import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, roc_curve, auc
from sklearn import metrics
import warnings
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import *
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
warnings.filterwarnings("ignore", category = FutureWarning)
warnings.filterwarnings("ignore", category = DeprecationWarning)


# Reading the data

data = pd.read_csv("sentiment.tsv", sep = "\t")
data.columns = ["label","text"]

# Converting label to numbers
le = LabelEncoder()
data["label"] = le.fit_transform(data["label"])

# Cleaning the data


def remove_pattern(input_text, pattern):
	r = re.findall(pattern,input_text)
	for i in r:
		input_text = re.sub(i, '',input_text)
	return input_text


# Removing twitter handles

data['tidy_tweet'] = np.vectorize(remove_pattern)(data["text"],"@[\w]*")

# Removing special characters, numbers and punctuations

data["tidy_tweet"] = data.tidy_tweet.str.replace("[^a-zA-Z#]", " ")

print("Before:" , data.head())

# Tokenizing the tweets 

tokenized_tweet = data['tidy_tweet'].apply(lambda x: x.split())

# Stemming

stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])
print(tokenized_tweet.head())

# Joining th tokenized word in the same data

for i in range(len(tokenized_tweet)):
	tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
data['tidy_tweet'] = tokenized_tweet

# Adding other column for length of the tweet and punctuation

# Getting the percentage of punctuations
def count_punctuation(text):
	count = sum([1 for char in text if char in string.punctuation])
	return round(count/(len(text) - text.count(" ")),3)*100

# Getting the length of the body
data["body_len"] = data["text"].apply(lambda x: len(x) - x.count(" "))
data["punct%"] = data["text"].apply(lambda x:count_punctuation(x))

print("After: ", data.head())

# Generating word cloud for data

all_words = ' '.join([text for text in data["tidy_tweet"]])
wordcloud = WordCloud(width = 800, height = 500, random_state = 21, max_font_size = 110	).generate(all_words)
plt.figure(figsize = (10,7))
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.show()

# Wordcloud for negative tweets

negative_words = ' '.join([text for text in data['tidy_tweet'][data['label']==0]])
wordcloud = WordCloud(width = 800, height = 500, random_state = 21, max_font_size = 110	).generate(negative_words)
plt.figure(figsize = (10,7))
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.show()

#	Feature Selection and Engineering
#	1. Count vectorizer
#	2. Tfidf vectorizer

bow_vectorizer = CountVectorizer(stop_words = 'english')
bow = bow_vectorizer.fit_transform(data['tidy_tweet'])
X_count_feat = pd.concat([data['body_len'],data['punct%'],pd.DataFrame(bow.toarray())],axis = 1)
print(X_count_feat.head())

tfidf_vectorizer = TfidfVectorizer(stop_words = 'english')
tfidf = tfidf_vectorizer.fit_transform(data['tidy_tweet'])
X_tfidf_feat = pd.concat([data['body_len'],data['punct%'],pd.DataFrame(tfidf.toarray())], axis = 1)

#Importing libraries for ML
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

model = []
model.append(('LR', LogisticRegression()))
model.append(('RF', RandomForestClassifier()))
model.append(('GB', GradientBoostingClassifier()))
model.append(('DT', DecisionTreeClassifier()))
model.append(('NB', GaussianNB()))
model.append(('KNN', KNeighborsClassifier()))
model.append(('SVC', SVC()))


# 10 cross fold validation fro all algos with X_count_feat 
for mod, clf in model:
	scores = cross_val_score(clf,X_count_feat, data["label"], scoring = 'accuracy', cv = 10)
	print("Model is %s and Score is %s" %(mod, scores.mean()))

# 10 cross fold validation fro all algos with X_tfidf_feat
for mod, clf in model:
	scores = cross_val_score(clf,X_tfidf_feat, data["label"], scoring = 'accuracy', cv = 10)
	print("Model is %s and Score is %s" %(mod, scores.mean()))

# Parameter tuning Log Regression Count Vectorizer
param_grid = {'C':[0.001, 0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(LogisticRegression(), param_grid, cv = 10)
grid.fit(X_count_feat,data['label'])

print(grid.best_estimator_)
# Parameter tuning Log Regression Tfidf Vectorizer