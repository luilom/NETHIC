from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import load_files
from sklearn.preprocessing import normalize
from sklearn.neural_network import MLPClassifier
from nltk.stem.snowball import EnglishStemmer
from sklearn.externals import joblib
import pickle
import sys
import os
import re


def stemmed_words_count(doc):
	stemmer = EnglishStemmer()
	analyzer = CountVectorizer(stop_words='english').build_analyzer()
	pattern1 = re.compile("(\W+|^)\d+(\W+|$)")
	pattern2 = re.compile("(\W+|^)\d+\s+")
	pattern3 = re.compile("(\W+|^)\d+\Z")
	pattern4 = re.compile("\A\d+(\W+|$)")
	pattern5 = re.compile("\s+\d+(\W+|$)")
	doc = pattern1.sub(" ", doc)
	doc = pattern2.sub(" ", doc)
	doc = pattern3.sub(" ", doc)
	doc = pattern4.sub(" ", doc)
	doc = pattern5.sub(" ", doc)
	#pattern = re.compile("\w*\d+\w*")
	doc = pattern.sub(" ", doc)
	return (stemmer.stem(w) for w in analyzer(doc))



def __normalizeFeature(dataset):
	dataset = dataset.astype(float)
	i = 0
	j = 0
	for row in dataset:
		j = 0
		size  = row.getnnz()
		for column in row:
			if dataset[i,j] != 0:
				result =  round((round(dataset[i,j],8)/size), 8)
				dataset[i,j] = round(result,8)
			j = j + 1
		i = i + 1
	return dataset

def start_training(datasetName,datasetsFolder,featureType,pathToSave):

	"""loading train  dataset """
	data_train = load_files(datasetsFolder+"/"+str(datasetName), encoding='latin1')
	print('data loaded')
	"""create dataset with vectorizer and normalize"""
	vectorizer = CountVectorizer(decode_error="replace",stop_words='english',analyzer=stemmed_words_count)
	


	"""vectorizer dataset train"""
	X_train = vectorizer.fit_transform(data_train.data)

	#if featureType == "normalize":
		#X_train = __normalizeFeature(X_train)

	print("n_samples: %d, n_features: %d" % X_train.shape)
	X_train = normalize(X_train)
	y_train = data_train.target


	""" get features Name from vectorizer """
	#feature_names = vectorizer.get_feature_names()

	"""Save vectorizer in file .pickle"""
	pickle.dump(vectorizer.vocabulary_,open(pathToSave+"/dictionaries_"+featureType+"/dict_"+str(datasetName)+".pkl","wb"))

	"""CONFIGURO LA RETE NEURALE E TRAINING DELLA RETE NEURALE"""
	numrip=0
	tol=0.005
	
	clf = MLPClassifier(activation='logistic',hidden_layer_sizes=(60,), max_iter=200, shuffle=True,solver='adam', tol=tol)
	clf.fit(X_train,y_train)
	currentScore = clf.score(X_train,y_train)
	print("Try: ",numrip+1," Current score: ",currentScore)
	
	while currentScore < 0.8 and numrip<2 :
		numrip = numrip + 1
		clf = MLPClassifier(activation='logistic',hidden_layer_sizes=(60,), max_iter=200, shuffle=True,solver='adam', tol=tol)
		clf.fit(X_train,y_train)
		currentScore = clf.score(X_train,y_train)
		tol = tol * 0.1
		print("Try: ",numrip+1," Current score: ",currentScore)

	print ("Score on training set: %0.8f" % clf.score(X_train, y_train))

	joblib.dump(clf,pathToSave+"/neural_networks_"+featureType+"/NN_"+str(datasetName))




datasets = sys.argv[1]
""" featureType can be 'count'  or  'normalize' """
featureType = sys.argv[2]
pathToSave = sys.argv[3]

if not os.path.exists(pathToSave+"/neural_networks_"+str(featureType)):
    os.makedirs(pathToSave+"/neural_networks_"+str(featureType))
if not os.path.exists(pathToSave+"/dictionaries_"+str(featureType)):
    os.makedirs(pathToSave+"/dictionaries_"+str(featureType))


dirs = os.listdir(datasets)
for file in dirs:
    dataset = str(file)
    print (dataset)
    start_training(dataset,datasets,featureType,pathToSave)




