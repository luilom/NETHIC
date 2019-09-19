import logging
import Utility
import numpy as np
from optparse import OptionParser
import sys
import time
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import metrics
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from io import StringIO
from sklearn.neural_network import MLPClassifier
from nltk.stem.snowball import EnglishStemmer
from sklearn.externals import joblib
import pickle
from copy import deepcopy
from sklearn.datasets import make_multilabel_classification
import json
from time import gmtime, strftime

def stemmed_words_count(doc):
	stemmer = EnglishStemmer()
	analyzer = CountVectorizer().build_analyzer()
	return (stemmer.stem(w) for w in analyzer(doc))

def stemmed_words_tfidf(doc):
	stemmer = EnglishStemmer()
	analyzer = TfidfVectorizer().build_analyzer()
	return (stemmer.stem(w) for w in analyzer(doc))


def classify(categoryToLoad,data_test,cutoff,neuralNetworks,dictionaries,text_embedded,model_type):

	categories = json.load(open("../categories.json"))
	if categoryToLoad in categories.keys():
		dictionary = dictionaries[categoryToLoad]
		vectorizer = CountVectorizer(decode_error="replace",stop_words='english',vocabulary=dictionary,analyzer=stemmed_words_count)
		if "doc2vec" in model_type:
			X_testGlobal = [np.concatenate((text_embedded,np.array(vectorizer.transform(data_test.data).toarray()[0])),axis = None)]
		else:
			X_testGlobal = vectorizer.transform(data_test.data)
		
		y_testGlobal = data_test.target
		"""print("n_samples: %d, n_features: %d" % X_test.shape)"""
		#X_test = normalize(X_testGlobal)
		X_test = X_testGlobal
		y_test = y_testGlobal

		""" get features Name from vectorizer """
		feature_names = vectorizer.get_feature_names()


		"""load Neural network"""
		startLoadNN = time.time()*1000
		"""clf = joblib.load("neural_networks/NN_"+str(categoryToLoad))"""
		clf = neuralNetworks[categoryToLoad]
		stopLoadNN = time.time()*1000
		"""print("Time to load NN "+str(categoryToLoad)+": "+str(stopLoadNN-startLoadNN))"""

		startToClassifier = time.time()*1000
		pred = clf.predict_proba(X_test)
		
		#pred = clf.predict(X_test)
		stopToClassifier = time.time()*1000
		"""print("Time to classifier "+str(categoryToLoad)+": "+str(stopToClassifier-startToClassifier))"""

		#threshold = (max(pred[0])+min(pred[0]))/2 
		toReturn = []
		
		totalscore = 0
		while totalscore < round(cutoff, 2):
			localscore = pred[0,np.argmax(pred[0,:])]
			category = categories[str(categoryToLoad)][np.argmax(pred[0,:])].lower()
			toReturn.append(Utility.Result(category,localscore))
			pred[0,np.argmax(pred[0,:])]=0
			totalscore+=localscore
		
		return toReturn

		



