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
from collections import defaultdict







# Protected tags that shouldn't appear anywhere in any gene names.
refgen_v3_tag = "refgen_v3="
refgen_v4_tag = "refgen_v3="
ncbi_tag = "ncbi="
uniprot_tag = "uniprot="



def add_tag(token, tag):
	if len(token) > 0:
		return("{}{}".format(tag, token))
	else:
		return("")




