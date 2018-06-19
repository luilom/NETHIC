import Classifier
import Utility
import pandas as pd
import numpy as np
import sys
import pickle
import operator
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import Bunch
import os
import re
from sklearn.externals import joblib
import json
import time
from nltk.stem.snowball import EnglishStemmer


def stemmed_words_count(doc):
	stemmer = EnglishStemmer()
	analyzer = CountVectorizer().build_analyzer()
	return (stemmer.stem(w) for w in analyzer(doc))

def stemmed_words_tfidf(doc):
	stemmer = EnglishStemmer()
	analyzer = TfidfVectorizer().build_analyzer()
	return (stemmer.stem(w) for w in analyzer(doc))



def clearFileName(filename):
    split = filename.split("/")
    pattern1 = re.compile('[0-9]')
    split[-1] = pattern1.sub("", split[-1])
    split[-1]=split[-1][0:-1]
    return split[-1]


#This method return for any leaf a correct parent 
def fromLeafToParent(leaf, categories, root):
    for parent in categories:
        if leaf in categories[parent]:
            if parent != root:
                return parent
            else:
                return leaf
    return None
    
def classification(data_test):
    data = []
    with open(data_test.filenames[i], 'rb') as f:
        data.append(f.read())
        data = [d.decode("latin1", "strict") for d in data]


    doc = Bunch(data=data,filenames=data_test.filenames[i],target=data_test.target[i])

    X_test = vectorizer.fit_transform(doc.data)
    pred = model.predict_proba(X_test)

    toReturn = []
    for k in range(0,1):
        category = categories[startCategory][np.argmax(pred[0,:])].lower()
        localscore = pred[0,np.argmax(pred[0,:])]
        pred[0,np.argmax(pred[0,:])] = 0
        toReturn.append(Utility.Result(category,localscore))

    return toReturn
    

def startTestNN(levels,startCategory,pathDataset,pathNN,pathDict,featureType,categories,rootCategorization):

    #E' una lista di liste. Ogni lista contenuta contiene la classe reale e quella cacolata per ogni singolo documento. cm serve per generare la confusion_matrix
    cm =list()
    
    """START"""
    numCategoryToTake = 1

    #print("Start to load Neural Networks and Dictionaries")

    model = joblib.load(pathNN+"/"+"NN_"+startCategory)
    dictionary = joblib.load(pathDict+"/"+"dict_"+startCategory+".pkl")
    
    #print("End to load Neural Networks and Dictionaries")

    vectorizer = CountVectorizer(decode_error="replace",stop_words='english',vocabulary=dictionary,analyzer=stemmed_words_count)

    
    """CARICO IL DATASET"""
    data_test = load_files("../datasets/"+pathDataset, encoding='latin1')
    globalResults = list()

    
    correctClassifications = 0
    documentTaked = 0
    toReturn = list()

    print("     "+startCategory)
    for i in range(0,len(data_test.data)):
        current = list()
        realCategory = data_test.filenames[i] 
        parentCategory = fromLeafToParent(clearFileName(realCategory),categories,"society")

        if parentCategory == startCategory or rootCategorization == True:
            documentTaked += 1

            data = []
            with open(data_test.filenames[i], 'rb') as f:
                data.append(f.read())
                data = [d.decode("latin1", "strict") for d in data]


            doc = Bunch(data=data,filenames=data_test.filenames[i],target=data_test.target[i])

            X_test = vectorizer.fit_transform(doc.data)
            pred = model.predict_proba(X_test)

            selectedCategory = []
            
            for k in range(0,numCategoryToTake):
                category = categories[startCategory][np.argmax(pred[0,:])].lower()
                localscore = pred[0,np.argmax(pred[0,:])]
                pred[0,np.argmax(pred[0,:])] = 0
                selectedCategory.append(Utility.Result(category,localscore))

            for i in range(len(selectedCategory)):
                if rootCategorization == False:
                    if selectedCategory[i].label == clearFileName(realCategory):
                        correctClassifications += 1
                        break
                else:
                    if selectedCategory[i].label == parentCategory:
                        correctClassifications += 1
                        break
            current.append(selectedCategory[i].label)#calcolato
            if rootCategorization:
                if parentCategory == "root":
                    current.append(clearFileName(realCategory.split("/")[-1]))#reale
                else:
                    current.append(parentCategory)#reale
            else:
                current.append(clearFileName(realCategory.split("/")[-1]))#reale

            cm.append(current)

    toReturn.append(startCategory)
    toReturn.append(float(correctClassifications)/float(documentTaked))
    toReturn.append(documentTaked)

    return toReturn,cm
	
        

startCategory = sys.argv[1]
pathDataset = sys.argv[2]
pathNNDICT = sys.argv[3]
featureType = sys.argv[4]
taxonomy = sys.argv[5]



#activationfunction = ['identity', 'logistic', 'tanh', 'relu']
#solvers = ['sgd','lbfgs','adam']
activationfunction = ['logistic']
solvers = ['adam']
categories = json.load(open(str(pathNNDICT+"/"+taxonomy+"_categories.json")))

if not os.path.exists(taxonomy):
    os.makedirs(taxonomy)


configurationResults = dict()
for function in activationfunction:
    for solver in solvers:
        print(function+"_"+solver)
        pathNN = pathNNDICT+"NN/"+taxonomy+"/"+function+"_"+solver+"/neural_networks_"+str(featureType)
        pathDict = pathNNDICT+"DICT/"+taxonomy+"/"+function+"_"+solver+"/dictionaries_"+str(featureType)
        toSave = list()
        avgScoreCurrentSetting = 0
        for category in categories:
            if category == startCategory:
                result, cm  = startTestNN(1,category,pathDataset,pathNN,pathDict,featureType,categories,True)
            else:
                result, cm = startTestNN(1,category,pathDataset,pathNN,pathDict,featureType,categories,False)

            toSave.append(result)

            avgScoreCurrentSetting += result[1]

            testAccuracyDataFrame = pd.DataFrame(cm, columns=['evaluated','real'])

            if not os.path.exists(taxonomy+"/confusion_matrix/"+function+"_"+solver):
                os.makedirs(taxonomy+"/confusion_matrix/"+function+"_"+solver)

            testAccuracyDataFrame.to_csv(taxonomy+"/confusion_matrix/"+function+"_"+solver+"/"+str(category)+".csv")
        
        configurationResults[function+"_"+solver] = avgScoreCurrentSetting/len(categories)

        testAccuracyDataFrame = pd.DataFrame(toSave, columns=['category','score','n_samples'])
        testAccuracyDataFrame.to_csv(taxonomy+"/test_accuracy_("+function+"_"+solver+").csv")

pd.DataFrame.from_dict(configurationResults,orient='index').to_csv(taxonomy+"/test_accuracy_singleNN.csv", sep=',')



