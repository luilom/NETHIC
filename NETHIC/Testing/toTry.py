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


def fromLeafToAncestor(leaf, categories, parent):
    if leaf in categories[parent]:
        if parent != "root":
            return parent
        else:
            return leaf
    elif leaf in categories["root"]:
        return leaf
    else:
        return fromLeafToAncestor(fromLeafToParent(leaf,categories,"root"),categories,parent)
    
    
    

#This method return for any leaf a correct parent 
def fromLeafToParent(leaf, categories, root):
    for parent in categories:
        if leaf in categories[parent]:
            if parent != root:
                return parent
            else:
                return leaf

    return None


categories = json.load(open(str("../dante_categories.json")))

parent = fromLeafToAncestor("other_religious_adversary",categories,"offensive_weapon")

print(parent)