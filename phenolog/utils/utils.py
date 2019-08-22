from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import DistanceMetric
from itertools import product
from scipy import spatial
from nltk.corpus import wordnet
import gensim
import numpy as np
import pandas as pd
import fastsemsim as fss
import string
import itertools
import pronto
import pickle
from collections import defaultdict




def function_wrapper(function, args):
	result = function(*args)
	return(result)




def save_to_pickle(obj, path):
	pickle.dump(obj, open(path,"wb"))




def load_from_pickle(path):
	obj = pickle.load(open(path,"rb"))
	return(obj)





