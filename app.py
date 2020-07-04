from flask import Flask, render_template, url_for, request
import pandas as pd
import numpy as np
from nltk.stem.porter import PorterStemmer
import re
import string
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

def remove_pattern(input_text, pattern):
	r = re.findall(pattern,input_text)
	for i in r:
		input_text = re.sub(i, '',input_text)
	return input_text

def count_punctuation(text):
	count = sum([1 for char in text if char in string.punctuation])
	return round(count/(len(text) - text.count(" ")),3)*100

app = Flask(__name__)

data = pd.read_csv("sentiment.tsv", sep = "\t")
data.columns = ["label","text"]

#features and labels
data["label"] = data["label"].map({'pos':0,'neg':1})
data['tidy_tweet'] = np.vectorize(remove_pattern)(data['text'],'@[\w]*')
data["tidy_tweet"] = data.tidy_tweet.str.replace("[^a-zA-Z#]", " ")
tokenized_tweet = data['tidy_tweet'].apply(lambda x: x.split())

# Stemming

stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])
for i in range(len(tokenized_tweet)):
	tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
data['tidy_tweet'] = tokenized_tweet
data["body_len"] = data["text"].apply(lambda x: len(x) - x.count(" "))
data["punct%"] = data["text"].apply(lambda x:count_punctuation(x))
X = data["tidy_tweet"]
y = data["label"]

cv = CountVectorizer(stop_words = 'english')
X = cv.fit_transform(X)
X = pd.concat([data['body_len'],data['punct%'],pd.DataFrame(X.toarray())],axis = 1)

clf = LogisticRegression(C = 0.1, class_weight = None, dual = False, fit_intercept = True,
	intercept_scaling = 1, l1_ratio = None, max_iter = 100, multi_class = 'auto', n_jobs = None, penalty = 'l2',
	random_state = None, solver = 'liblinear', tol = 0.0001, verbose = 0, warm_start = False)

clf.fit(X,y)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict', methods = ['POST'])
def predict():
	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = pd.DataFrame(cv.transform(data).toarray())
		body_len = pd.DataFrame([len(data)- data.count(" ")])
		punct = pd.DataFrame([count_punctuation(data)])
		total_data = pd.concat([body_len,punct,vect], axis = 1)
		my_pred = clf.predict(total_data)
	return render_template('result.html', prediction = 	my_pred)

if __name__ == '__main__':
	app.run(port = 4000)
