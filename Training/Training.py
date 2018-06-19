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

def start_training(dataset,featureType,function,solver,path,taxonomy):

	folderNNCurrentSetting = path+"NN/"+taxonomy+"/"+function+"_"+solver
	folderDictCurrentSetting = path+"DICT/"+taxonomy+"/"+function+"_"+solver

	if not os.path.exists(folderNNCurrentSetting):
		os.makedirs(folderNNCurrentSetting)
	if not os.path.exists(folderNNCurrentSetting+"/neural_networks_"+str(featureType)):
		os.makedirs(folderNNCurrentSetting+"/neural_networks_"+str(featureType))

	if not os.path.exists(folderDictCurrentSetting):
		os.makedirs(folderDictCurrentSetting)
	if not os.path.exists(folderDictCurrentSetting+"/dictionaries_"+str(featureType)):
		os.makedirs(folderDictCurrentSetting+"/dictionaries_"+str(featureType))

	toReturn = dict()
	"""loading train  dataset """
	
	print('data loaded')
	"""create dataset with vectorizer and normalize"""
	vectorizer = CountVectorizer(decode_error="replace",stop_words='english',analyzer=stemmed_words_count)
	

	"""vectorizer dataset train"""
	X_train = vectorizer.fit_transform(dataset.data)

	if featureType == "normalize":
		X_train = __normalizeFeature(X_train)
	
	n_samples, n_features = X_train.shape
	toReturn["n_samples"] = n_samples
	toReturn["n_features"] = n_features

	print("n_samples: %d, n_features: %d" % X_train.shape)
	X_train = normalize(X_train)
	y_train = dataset.target


	""" get features Name from vectorizer """
	feature_names = vectorizer.get_feature_names()

	"""Save vectorizer in file .pickle"""
	pickle.dump(vectorizer.vocabulary_,open(folderDictCurrentSetting+"/dictionaries_"+str(featureType)+"/dict_"+str(datasetName)+".pkl","wb"))

	"""CONFIGURO LA RETE NEURALE E TRAINING DELLA RETE NEURALE"""
	numrip=0
	tol=0.005
	max_iter = 200

	clf = MLPClassifier(activation=function,hidden_layer_sizes=(60,), max_iter=max_iter, shuffle=True,solver=solver, tol=tol)
	clf.fit(X_train,y_train)
	currentScore = clf.score(X_train,y_train)
	print("Try: ",numrip+1," Current score: ",currentScore)
	
	while currentScore < 0.8 and numrip<1 :
		numrip = numrip + 1
		clf = MLPClassifier(activation=function,hidden_layer_sizes=(60,), max_iter=max_iter, shuffle=True,solver=solver, tol=tol)
		clf.fit(X_train,y_train)
		currentScore = clf.score(X_train,y_train)
		tol *= 0.1
		max_iter += 50
		print("Try: ",numrip+1," Current score: ",currentScore)
		
	print ("Score on training set: ",currentScore)

	joblib.dump(clf,folderNNCurrentSetting+"/neural_networks_"+featureType+"/NN_"+str(datasetName))
	toReturn["score"] = currentScore 
	return toReturn


datasetsFolder = sys.argv[1]
""" featureType can be 'count'  or  'normalize' """
featureType = sys.argv[2]
path = sys.argv[3]
taxonomy = sys.argv[4]

if not os.path.exists(path+"/NN/"+taxonomy+"/neural_networks_"+str(featureType)):
    os.makedirs(path+"/NN/"+taxonomy+"/neural_networks_"+str(featureType))
if not os.path.exists(path+"/DICT/"+taxonomy+"/dictionaries_"+str(featureType)):
    os.makedirs(path+"/DICT/"+taxonomy+"/dictionaries_"+str(featureType))

activationfunction = ['logistic']
solvers = ['adam']
#activationfunction = ['logistic']
#solvers = ['adam']
dirs = os.listdir(datasetsFolder)
AVGResultsWithDifferentConfigurations = dict()
for function in activationfunction:
	for solver in solvers:
		listTrainingAccuracy = list()
		localScore = 0
		print(function+" - "+solver)
		for file in dirs:
			trainingAccuracy = list()
			datasetName = str(file)
			print(datasetName)
			dataset = load_files(datasetsFolder+"/"+str(datasetName), encoding='latin1')
			result = start_training(dataset,featureType,function,solver,path,taxonomy)
			trainingAccuracy.append(datasetName)
			trainingAccuracy.append(result['score'])
			localScore += result['score']
			trainingAccuracy.append(result['n_samples'])
			trainingAccuracy.append(result['n_features'])
			listTrainingAccuracy.append(trainingAccuracy)

		AVGResultsWithDifferentConfigurations[str(function)+"-"+str(solver)] = localScore/len(dirs)

		trainingAccuracyDataFrame = pd.DataFrame(listTrainingAccuracy, columns=['category','score','n_samples','n_features'])
		trainingAccuracyDataFrame.set_index("category")
		if not os.path.exists(taxonomy):
			os.makedirs(taxonomy)

		trainingAccuracyDataFrame.to_csv(taxonomy+"/training_accuracy_"+str(function+"_"+solver)+".csv")

		print(AVGResultsWithDifferentConfigurations)

		pd.DataFrame.from_dict(AVGResultsWithDifferentConfigurations,orient='index').to_csv(taxonomy+"/training_accuracy_"+taxonomy+".csv", sep=',')