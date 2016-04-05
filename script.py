import pandas as pd
from bs4 import BeautifulSoup
import nltk
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import pickle
from sklearn import cross_validation

train = pd.read_csv("data.tsv", header = 0, delimiter = "\t", quoting = 3)



clean_train = train


def reviewToWords(review):
    #HTML removed
    review_text = BeautifulSoup(review).get_text()
    
    #Non-letters removed
    only_letters = re.sub("[^a-zA-Z]", " ", review_text) 
    
    #Convert to lower case and split
    words = only_letters.lower().split()
    
    #Convert stop words to a set because its faster
    stops = set(stopwords.words("english"))
    
    #Remove stop words
    req_words = [w for w in words if not w in stops]
    
    #Join words separated by space and return
    return(' '.join(req_words))
    
count = train["review"].size


clean_list = []

for i in xrange(count) :
    clean_train["review"][i] = reviewToWords(train["review"][i])
    clean_list.append(reviewToWords(train["review"][i]))
    
clean_train.to_csv("cleanReviews.tsv", delimiter = "\t", sep = '\t')    


#creating vocabulary
vectorizer = CountVectorizer(analyzer = "word", min_df = 2)
trainDataFeatures = vectorizer.fit_transform(clean_list)
trainDataFeatures = trainDataFeatures.toarray()

vocabulary = vectorizer.get_feature_names()

dist = np.sum(trainDataFeatures, axis = 0)

f = open('vocabulary.txt' , 'w')
for v in vocabulary:
    f.write(v + '\n')
    
f.close()


#training and saving a naive bayes model
clf = MultinomialNB()
clf.fit(trainDataFeatures, train["sentiment"])

f = open('model.p', 'w')
pickle.dump(clf, f)
f.close()

#10 fold cross validation
X = trainDataFeatures
y = train["sentiment"]

skf = cross_validation.StratifiedKFold(y, n_folds = 10)
for train_index, test_index in skf:
	print "TRAIN: ",train_index
	print "TEST: ", test_index

scores = cross_validation.cross_val_score(clf, X, y, cv=skf)
print "Scores : ",scores
print("Average Accuracy: %f" % (scores.mean()))
