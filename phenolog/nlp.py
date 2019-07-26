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
from fuzzywuzzy import fuzz
from fuzzywuzzy import process




def get_clean_description(description):
	description = remove_punctuation(description)
	description = description.lower()
	return(description)

def get_clean_token_list(description):
	description = remove_punctuation(description)
	token_list = description.lower().split()
	return(token_list)

def remove_punctuation(text):
	translator = str.maketrans('', '', string.punctuation)
	return(text.translate(translator))




def append_related_words(description, related_word_list):
	combined_list = [description]
	combined_list.extend(related_word_list)
	description = " ".join(combined_list).strip()
	return(description)







def load_word2vec_model(model_path):
	model = gensim.models.word2vec.Word2Vec.load(model_path)
	return(model)

def train_word2vec_model(model_path, training_textfile_path, size=300, sg=1, hs=1, sample=1e-3, window=10, alpha=0.025, workers=5):
	text = gensim.models.word2vec.Text8Corpus(training_textfile_path)
	model = gensim.models.word2vec.Word2Vec(sentences, size=size, sg=sg, hs=hs, sample=sample, window=window, alpha=alpha, workers=workers) 
	model.save(model_path)















def binary_search_rabin_karp(pat, txt, q): 
	"""
	Searches for exact matches to a pattern in a longer string (fast). 
	Adapted from implementation by Bhavya Jain found at
	https://www.geeksforgeeks.org/rabin-karp-algorithm-for-pattern-searching/
	Args:
		pat (str): The shorter text to search for.
		txt (str): The larger text to search in.
		q (int): A prime number that is used for hashing.
	Returns:
		boolean: True if the pattern was found, false is it was not.
	"""

	# Make sure the pattern is smaller than the text.
	if len(pat)>len(txt):
		return(False)

	d = 256					# number of characters in vocabulary
	M = len(pat) 
	N = len(txt) 
	i = 0
	j = 0
	p = 0    				# hash value for pattern 
	t = 0    				# hash value for txt 
	h = 1

	found_indices = []

	for i in range(M-1): 
		h = (h * d)% q 
	for i in range(M): 
		p = (d * p + ord(pat[i]))% q 
		t = (d * t + ord(txt[i]))% q 
	for i in range(N-M + 1): 
		if p == t: 
			for j in range(M): 
				if txt[i + j] != pat[j]: 
					break
			j+= 1
			if j == M: 
				# Pattern found at index i.
				# found_indices.append(i)
				return(True)

		if i < N-M: 
			t = (d*(t-ord(txt[i])*h) + ord(txt[i + M]))% q 
			if t < 0: 
				t = t + q 

	# Pattern was never found.			
	return(False)








def binary_search_fuzzy(pat, txt, threshold, local=1):
	"""Searches for fuzzy matches to a pattern in a longer string (slow).
	Args:
		pat (str): The shorter text to search for.
		txt (str): The larger text to search in.
		threshold (int): Value between 0 and 1 at which matches are considered real.
		local (int, optional): Alignment method, 0 for global 1 for local.
	Returns:
		boolean: True if the pattern was found, false if it was not.
	"""
	similarity_score = 0.000
	if local==1:
		similarity_score = fuzz.partial_ratio(pat, txt)
	else:
		similarity_score = fuzz.ratio(pat, txt)
	if similarity_score >= threshold*100:
		return(True)
	return(False)




def occurences_search_fuzzy(patterns, txt, threshold, local=1):
	"""
	Searches for occurences of any of the patterns in the longer string (slow).
	The method process.extractBests() returns a list of tuples where the first
	item is the pattern string and the second item is the alignment score for 
	that pattern.
	
	Args:
		patterns (list): The shorter text strings to search for.
		txt (str): The larger text to search in.
		threshold (int): Value between 0 and 1 at which matches are considered real.
		local (int, optional): Alignment method, 0 for global 1 for local.
	Returns:
		list: A sublist of the patterns argument containing only the found strings.
	"""
	patterns_found = []
	threshold = threshold*100
	if local==1:
		method = fuzz.partial_ratio
	else:
		method = fuzz.ratio
	best_matches = process.extractBests(query=txt, choices=patterns, scorer=method, score_cutoff=threshold)
	patterns_found = [match[0] for match in best_matches]
	return(patterns_found)

































