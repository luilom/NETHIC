import Classifier
import Result
import pandas
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

def stemmed_words_count(doc):
	stemmer = EnglishStemmer()
	analyzer = CountVectorizer().build_analyzer()
	return (stemmer.stem(w) for w in analyzer(doc))

def stemmed_words_tfidf(doc):
	stemmer = EnglishStemmer()
	analyzer = TfidfVectorizer().build_analyzer()
	return (stemmer.stem(w) for w in analyzer(doc))


def recursiveClassification(startCategory,count,maxLevel,dictResults,i,data_test,cutoff,neuralNetworks,dictionaries):   
    result = recursiveClassificationTask(startCategory,count,maxLevel,i,data_test,cutoff,neuralNetworks,dictionaries)
    if result is not None:
        dictResults[startCategory] = result
        for result in dictResults[startCategory]:
            recursiveClassification(dictResults[startCategory][result].target,count + 1,maxLevel,dictResults,i,data_test,cutoff,neuralNetworks,dictionaries)
        return dictResults
    return None


def recursiveClassificationTask(startCategory,currLevel,maxLevel,i,data_test,cutoff,neuralNetworks,dictionaries):
    isRoot = False
    if startCategory == "root":
        isRoot = True
    if int(currLevel) <= int(maxLevel):
        dictScores = dict()
        categories =  Classifier.classify(startCategory,data_test,cutoff,neuralNetworks,dictionaries)
        if categories is not None:
            for i in range(len(categories)):
                dictScores[str(categories[i].target)] = categories[i]
            return dictScores
    return None   
    


def computeResult(category,dictResults,dictScorePath,parents):
    if category in dictResults.keys():
        for k in dictResults[category].keys():          
            element = dictResults[category][k]
            if parents is None:
                parents = []            
            parents.append(element)
            parents = computeResult(element.target,dictResults,dictScorePath,parents)              
    else:
        path = ""
        scorePath = float(0)
        for i in range(len(parents)):
            path+=str(parents[i].target)+"/"
            scorePath = float(scorePath + float(parents[i].score))
        scorePath = float(scorePath/len(parents))
        dictScorePath[path] = scorePath
    return parents[0:len(parents)-1]



def computeResult2(category,dictResults,dictScorePath,parents):
    if category in dictResults.keys():
        for k in dictResults[category].keys():          
            element = dictResults[category][k]
            if parents is None:
                parents = []            
            parents.append(element)
            parents = computeResult2(element.target,dictResults,dictScorePath,parents)              
    else:
        path = ""
        scorePath = float(0)
        for i in range(len(parents)):
            path+=str(parents[i].target)+"/"
            scorePath = float(scorePath + float(parents[i].score/(i+1)))
        scorePath = float(scorePath/len(parents))
        dictScorePath[path] = scorePath
    return parents[0:len(parents)-1]

def clearFileName(filename):
    split = str(data_test.filenames[i]).split("/")
    pattern1 = re.compile('[0-9]')
    split[-1] = pattern1.sub("", split[-1])
    split[-1]=split[-1][0:-1]
    return split[-1]


#This method return for any leaf a correct parent 
def fromLeafToParent(leaf, categories):
    for parent in categories:
        if leaf in categories[parent]:
            if parent != "root":
                return parent
            else:
                return leaf


"""START"""

levels = sys.argv[1]
startCategory = sys.argv[2]
datasetName = sys.argv[3]
path = sys.argv[4]
featureType = sys.argv[5]


pathNN = path+'/neural_networks_'+str(featureType)
pathDict = path+'/dictionaries_'+str(featureType)
categories = json.load(open("categories.json"))


neuralNetworks = dict()
dictionaries = dict()

print("Start to load Neural Networks and Dictionaries")

"""CARICO LE RETI NEURALI"""
for filename in os.listdir(pathNN):
    neuralNetworks[filename] = joblib.load(pathNN+"/"+filename)

"""CARICO I DIZIONARI"""
for filename in os.listdir(pathDict):
    dictionaries[filename.replace(".pkl","")] = joblib.load(pathDict+"/"+filename)

print("End to load Neural Networks and Dictionaries")




"""CARICO IL DATASET"""
data_test = load_files(datasetName, encoding='latin1')
globalResults = list()


for cut in range(0,225,25):
    print("CUT: ---> "+str(cut))
    currentResult = list()
    numDocumentTaked = 0
    numCorrectDocument = 0
    confusionMatrixList = list()
    

    #CLASSIFICO UN DOCUMENTO ALLA VOLTA con il CUT Scelto
    for i in range(0,len(data_test.data)):
        currentConfusionMatrixResult = list()
        if len(data_test.data[i].split(" ")) >= cut:
            numDocumentTaked += 1
            
            cutoff = round(0.8, 2)
            maxIter = 4
            tollerance = 0.7
            currentTollerance = 0
            data = []
            with open(data_test.filenames[i], 'rb') as f:
                data.append(f.read())
                data = [d.decode("latin1", "strict") for d in data]


            newData_test = Bunch(data=data,filenames=data_test.filenames[i],target=data_test.target[i])
            
            while maxIter>0 and currentTollerance<tollerance:
                dictResults = dict()
                dictResults = recursiveClassification(startCategory,0,levels,dictResults,i,newData_test,round(cutoff, 2),neuralNetworks,dictionaries)
                cutoff = cutoff - round(0.1, 2)
                maxIter = maxIter-1
                results = dict()
                computeResult(startCategory,dictResults,results,[])
                sorted_results = sorted(results.items(), key=operator.itemgetter(1),reverse=True)
                currentTollerance = 0
                maxToNormalize = sorted_results[0][1]
                for sorted_res in sorted_results:
                    currentTollerance=currentTollerance+(sorted_res[1]/maxToNormalize)
            
                currentTollerance = currentTollerance/len(sorted_results)

            filename = clearFileName(data_test.filenames[i])
            #print("FILE NAME: ",filename)
            #print("CUTOFF: ",round(cutoff, 2))
            #print("Tollerance: ",currentTollerance)
            
            numCategoryToResults = 0

            if len(sorted_results)>2:
                numCategoryToResults = 3
            else:
                numCategoryToResults = len(sorted_results)

            insert = False
            for j in range(0,numCategoryToResults):
                if str(filename) in sorted_results[j][0]:
                    numCorrectDocument += 1
                    insert = True
                    currentConfusionMatrixResult.append(fromLeafToParent(filename,categories))
                    break

            if insert == False:
                currentConfusionMatrixResult.append(sorted_results[0][0].split("/")[0])

            currentConfusionMatrixResult.append(fromLeafToParent(filename,categories))
            confusionMatrixList.append(currentConfusionMatrixResult)
           # print("FILE NAME: ",filename)

            if numCorrectDocument > 0:
                print("Accuracy: "+str(float(numCorrectDocument)/float(numDocumentTaked))+" %"+"       /         Progress: "+str(i))
                

            #print("*************************************************************")
        
    currentResult.append(cut)
    currentResult.append(float(numCorrectDocument)/float(numDocumentTaked))
    currentResult.append(numCorrectDocument)
    currentResult.append(numDocumentTaked)
    globalResults.append(currentResult)


    
    confuzionMatrixListdataframe = pandas.DataFrame(confusionMatrixList,columns=["predicted","expected"])
    confuzionMatrixListdataframe.to_csv("confusionMatrix_"+str(cut)+".csv")

dataframeAccuracy = pandas.DataFrame(globalResults, columns=['cut','score','#Correct_document','#Total_document'])
dataframeAccuracy.set_index("cut")
dataframeAccuracy.to_csv("test_accuracy_global.csv")

    

    