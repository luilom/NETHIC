from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import load_files
from sklearn.preprocessing import normalize
from sklearn.neural_network import MLPClassifier
from nltk.stem.snowball import EnglishStemmer
from sklearn.externals import joblib
import pickle
import sys
import os
import string
import re
import pandas as pd


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

def start_training(datasetName,datasetsFolder,featureType,function,solver):
	toReturn = dict()
	"""loading train  dataset """
	data_train = load_files(datasetsFolder+"/"+str(datasetName), encoding='latin1')
	print('data loaded')
	"""create dataset with vectorizer and normalize"""
	vectorizer = CountVectorizer(decode_error="replace",stop_words='english',analyzer=stemmed_words_count)
	

	"""vectorizer dataset train"""
	X_train = vectorizer.fit_transform(data_train.data)

	if featureType == "normalize":
		X_train = __normalizeFeature(X_train)
	
	n_samples, n_features = X_train.shape
	toReturn["n_samples"] = n_samples
	toReturn["n_features"] = n_features

	print("n_samples: %d, n_features: %d" % X_train.shape)
	X_train = normalize(X_train)
	y_train = data_train.target


	""" get features Name from vectorizer """
	feature_names = vectorizer.get_feature_names()

	"""Save vectorizer in file .pickle"""
	#pickle.dump(vectorizer.vocabulary_,open("dictionaries_"+featureType+"/dict_"+str(datasetName)+".pkl","wb"))

	"""CONFIGURO LA RETE NEURALE E TRAINING DELLA RETE NEURALE"""
	numrip=0
	tol=0.005
	
	clf = MLPClassifier(activation=function,hidden_layer_sizes=(60,), max_iter=200, shuffle=True,solver=solver, tol=tol)
	clf.fit(X_train,y_train)
	currentScore = clf.score(X_train,y_train)
	print("Try: ",numrip+1," Current score: ",currentScore)
	
	while currentScore < 0.8 and numrip<1 :
		numrip = numrip + 1
		clf = MLPClassifier(activation=function,hidden_layer_sizes=(60,), max_iter=200, shuffle=True,solver=solver, tol=tol)
		clf.fit(X_train,y_train)
		currentScore = clf.score(X_train,y_train)
		tol = tol * 0.1
		print("Try: ",numrip+1," Current score: ",currentScore)
		
	print ("Score on training set: ",currentScore)

	#joblib.dump(clf,"neural_networks_"+featureType+"/NN_"+str(datasetName))
	toReturn["score"] = currentScore 
	return toReturn


datasets = sys.argv[1]
""" featureType can be 'count'  or  'normalize' """
featureType = sys.argv[2]

if not os.path.exists("neural_networks_"+str(featureType)):
    os.makedirs("neural_networks_"+str(featureType))
if not os.path.exists("dictionaries_"+str(featureType)):
    os.makedirs("dictionaries_"+str(featureType))

activationfunction = ['identity', 'logistic', 'tanh', 'relu']
solvers = ['sgd','lbfgs','adam']
dirs = os.listdir(datasets)
resultsWithDifferentConfigurations = dict()
for function in activationfunction:
	for solver in solvers:
		listTrainingAccuracy = list()
		localScore = 0
		print(function+" - "+solver)
		for file in dirs:
			trainingAccuracy = list()
			dataset = str(file)
			print (dataset)
			result = start_training(dataset,datasets,featureType,function,solver)
			trainingAccuracy.append(dataset)
			trainingAccuracy.append(result['score'])
			localScore += result['score']
			trainingAccuracy.append(result['n_samples'])
			trainingAccuracy.append(result['n_features'])
			listTrainingAccuracy.append(trainingAccuracy)

		resultsWithDifferentConfigurations[str(function)+"-"+str(solver)] = localScore/len(dirs)

		trainingAccuracyDataFrame = pd.DataFrame(listTrainingAccuracy, columns=['category','score','n_samples','n_features'])
		trainingAccuracyDataFrame.set_index("category")
		trainingAccuracyDataFrame.to_csv("training_accuracy_"+str(function+"_"+solver)+".csv")

		print(resultsWithDifferentConfigurations)



		pd.DataFrame.from_dict(resultsWithDifferentConfigurations,orient='index').to_csv("settingResults.csv", sep=',')