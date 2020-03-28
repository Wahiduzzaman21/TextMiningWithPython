#import packages
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import GridSearchCV
import spacy
from nltk.corpus import stopwords
import string
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')

x_train= newsgroups_train.data
y_train = newsgroups_train.target

x_test = newsgroups_test.data
y_test = newsgroups_test.target

# pre processing

# lemmatization
nlp = spacy.load("en_core_web_sm")
x_train_nlp = [[x.lemma_ for x in nlp(y)]for y in x_train]
x_test_nlp = [[x.lemma_ for x in nlp(y)]for y in x_test]

# remove stop words
stop_en = stopwords.words("English")

x_cleaned_stopword= []
for x in x_train_nlp:
    x_cleaned_stopword.append([y for y in x if not y in stop_en])

x_cleaned_stopword_test= []
for x in x_test_nlp:
    x_cleaned_stopword_test.append([y for y in x if not y in stop_en])

# remove punctuation
x_cleaned_stopword_punctuation = []
for x in x_cleaned_stopword:
    x_cleaned_stopword_punctuation.append([y for y in x if not y in list(string.punctuation)])

x_cleaned_stopword_punctuation_test = []
for x in x_cleaned_stopword_test:
    x_cleaned_stopword_punctuation_test.append([y for y in x if not y in list(string.punctuation)])

# remove pronoun
useless = ["-PROUN-"]
x_cleaned_stopword_punctuation_pronoun = []
for x in x_cleaned_stopword_punctuation:
    x_cleaned_stopword_punctuation_pronoun.append([y for y in x if not y in useless])

x_cleaned_stopword_punctuation_pronoun_test = []
for x in x_cleaned_stopword_punctuation_test:
    x_cleaned_stopword_punctuation_pronoun_test.append([y for y in x if not y in useless])


# remove \n and dash
x_cleaned_stopword_punctuation_pronoun_newline_dash = []
for x in x_cleaned_stopword_punctuation_pronoun:
    x_cleaned_stopword_punctuation_pronoun_newline_dash.append([y for y in x if not ("--" in y or '\n' in y)])

x_cleaned_stopword_punctuation_pronoun_newline_dash_test = []
for x in x_cleaned_stopword_punctuation_pronoun_test:
    x_cleaned_stopword_punctuation_pronoun_newline_dash_test.append([y for y in x if not ("--" in y or '\n' in y)])

x_clean_data =["".join(y) for y in x_cleaned_stopword_punctuation_pronoun_newline_dash]
x_clean_data_test =["".join(y) for y in x_cleaned_stopword_punctuation_pronoun_newline_dash_test]

# Feature representation
vec = TfidfVectorizer()
x_train_vec = vec.fit_transform(x_clean_data)
x_test_vec = vec.fit_transform(x_clean_data_test)

# Apply algorithm
clf = LinearSVC(C=1, multi_class='ovr',dual=True)
clf.fit(x_train_vec,y_train)

# Validation
y_predict = clf.predict(x_test_vec)

print("Accuracy: ",accuracy_score(y_pred=y_predict,y_true=y_test))
print("Precision: ",precision_score(y_pred=y_predict,y_true=y_test,average="micro"))
print("Recall: ",recall_score(y_pred=y_predict,y_true=y_test,average="micro"))
print("F1: ",f1_score(y_pred=y_predict,y_true=y_test,average="micro"))


def showtop10(classifier,vectorizer,categories):
    feature_names=np.asarray(vectorizer.get_feature_names())
    for i,category in enumerate(categories):
        top10=np.argsort(classifier.coef_[i])[-10]
        print("%s: %s" % (category," ".join(feature_names[top10])))

showtop10(clf,vec,newsgroups_train.target_names)


# Parameter Tuning

parameters = {'C':[0.01,0.05,0.1,0.25,0.5,0.75,1,1.5,2],"dual":[True,False]}
clf = LinearSVC()
grid= GridSearchCV(clf,parameters)
grid.fit(x_train_vec,y_train)

print("Grid Best Parameter: ",grid.best_params_)
print("Grid Best Score: ",grid.best_score_)