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









def add_prefix(token, prefix):
	"""
	Attaches the passed in prefix argument to the front of the token,
	unless the token is an empty string in which case nothing happens
	(avoids accidentally making a meaningless token ("") meaningful by
	modifying it).
	Args:
	    token (str): Any string.
	    prefix (str): Any string.
	Returns:
	    str: The token with the prefix added to the beginning.
	"""
	if len(token) > 0:
		return("{}{}".format(prefix, token))
	else:
		return("")






def save_to_pickle(obj, path):
	pickle.dump(obj, open(path,"wb"))




def load_from_pickle(path):
	obj = pickle.load(open(path,"rb"))
	return(obj)





