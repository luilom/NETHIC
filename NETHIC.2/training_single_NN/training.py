import json
import sys
import logging

training_type = sys.argv[1]

if "doc2vec-bow" in training_type:
    dataframe_folder = "dataframes_doc2vec-BOW"
    folder_to_save_NN = "results_single_NN_doc2vec-BOW"
    logger_file = "logger_doc2vec-BOW.log"
    neural_networks_folder = "neural_networks/bow-doc2vec_NNs"
elif "doc2vec" in training_type:
    dataframe_folder = "dataframes_doc2vec"
    folder_to_save_NN = "results_single_NN_doc2vec"
    logger_file = "logger_doc2vec.log"
    neural_networks_folder = "neural_networks/doc2vec_NNs"
else:
    dataframe_folder = "dataframes_BOW"
    folder_to_save_NN = "results_single_NN_BOW"
    logger_file = "logger_BOW.log"
    neural_networks_folder = "neural_networks/bow_NNs"


#filename=logger_file,
logging.basicConfig(filename=logger_file,level=logging.INFO, format='%(asctime)-15s %(levelname)s %(filename)s %(message)s')
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
import numpy as np
import tqdm
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import preprocessing


logging.info("Loading categories from json file")
categories = json.load(open("../categories.json"))

logging.info("Reading types of training between doc2vec or doc2vec-bow")

tol=0.005
max_iter = 200
kf = KFold(n_splits=5)

results_all_categories = dict()

for category in categories:
    logging.info("Loading dataframe for current category : {}".format(category))
    dataset = pickle.load(open('../datasetBuilder/'+dataframe_folder+'/single_categories/'+category+'.pkl', 'rb'))
    X = dataset["vector"].tolist()
    X = [d.tolist() for d in X]
    y = dataset["label"].tolist()
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    logging.info("Dataset normalization")
    #X = preprocessing.normalize(X)
    logging.info("Splitting dataset in train and test")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
    X = None
    accuracy_training_k_fold_cv = list()
    counter = 0

    #Training in K-Fold
    for train_index, test_index in kf.split(X_train):
        counter += 1
        logging.info("Cross validation fold : {}".format(counter))
        logging.info("Splitting dataset in train and test for current cross validation")
        current_X_train, current_X_test = np.array(X_train)[train_index], np.array(X_train)[test_index]
        current_y_train, current_y_test = y_train[train_index], y_train[test_index]
        clf = MLPClassifier(activation='logistic',hidden_layer_sizes=(60,), max_iter=max_iter, shuffle=True,solver='adam', tol=tol, verbose=True)
        logging.info("Start Training")
        clf.fit(current_X_train,current_y_train)
        currentScore = clf.score(current_X_test,current_y_test)
        accuracy_training_k_fold_cv.append(currentScore)
        current_X_train = None 
        current_X_test = None
        current_y_train = None
        current_y_test = None

    #Training with all train dataset
    clf = MLPClassifier(activation='logistic',hidden_layer_sizes=(60,), max_iter=max_iter, shuffle=True,solver='adam', tol=tol, verbose=True)
    logging.info("Training model on training dataset")
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)

    logging.info("Start to test model and evaluate scores")
    current_results = dict()
    current_results["avg_accuracy_training_test_k_fold_cv"] = sum(accuracy_training_k_fold_cv) / len(accuracy_training_k_fold_cv)
    current_results["training_accuracies"] = clf.score(X_train,y_train)
    current_results["test_accuracies"] = clf.score(X_test,y_test)
    current_results["f1_score"] = f1_score(y_test, y_pred, average=None)
    current_results["precision"] = precision_score(y_test, y_pred, average=None)
    current_results["recall"] = recall_score(y_test, y_pred, average=None)

    logging.info("Saving score on file : {}".format(folder_to_save_NN+'/'+str(category)+'_results.pkl'))
    output1 = open(folder_to_save_NN+'/'+str(category)+'_results.pkl', 'wb')
    pickle.dump(current_results, output1)
    output1.close()
    logging.info("CURRENT RESULTS FOR CATEGORY : {} are \n {}".format(category,current_results))
    results_all_categories[category] = current_results
    pickle.dump(clf,open(neural_networks_folder+"/NN_"+str(category)+".pkl","wb"))
    print(current_results)


logging.info("Save results all categories on file {}".format(folder_to_save_NN+'/results_all_categories.pkl'))
output2 = open(folder_to_save_NN+'/results_all_categories.pkl', 'wb')
pickle.dump(results_all_categories, output2)
output2.close()
