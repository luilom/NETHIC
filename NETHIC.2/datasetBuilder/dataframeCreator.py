import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)
import os
from tqdm import tqdm
from gensim.models.doc2vec import Doc2Vec
from nltk.tokenize import word_tokenize

logging.info("Loading Doc2Vec model")
model3DBOW = Doc2Vec.load("enwiki_dbow/doc2vec.bin")

path_dataset_train = 'datasets/datasets_training_test_singole_reti'
#path_dataset_hierarchical_model = 'datasets/datasets_test_modello_gerarchico'
datasets = dict()

logging.info("Primo livello, vengono selezionati tutti i dataset da trasformare in dataframe")
for sub_folder in os.listdir(path_dataset_train):
    if os.path.isdir(os.path.join(path_dataset_train,sub_folder)):
        datasets[sub_folder] = os.path.join(path_dataset_train,sub_folder)

logging.info("Start to create dataframe for any dataset")
for key in tqdm(datasets.keys()):
    for x in os.walk(datasets[key]):
        currentText = open(dataset[key]+"/"+x).readlines()
        currentVector = model3DBOW.infer_vector(word_tokenize(currentText))
        print(currentVector)


    
