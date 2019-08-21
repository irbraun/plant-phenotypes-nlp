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






def remove_newlines(text):
	text = text.replace("\n", " ")
	text = text.replace("\t", " ")
	text = text.replace("\r", " ")
	return(text)







def add_end_tokens(description):
	if len(description) > 0:
		last_character = description[len(description)-1]
		end_tokens = [".", ";"]
		if not last_character in end_tokens:
			description = description+"."
	return(description)





def concatenate_descriptions(*descriptions):
	"""
	Combines multiple description strings into a single string. Characters which
	denote the end of a sentence or fragment are added where necessary so that the
	combined description will still be parseable with some other NLP package or 
	functions if generating a sentence tree or something like that.
	Args:
	    *descriptions: Description
	Returns:
	    TYPE: Description
	"""
	descriptions = [add_end_tokens(description) for description in descriptions]
	description = " ".join(descriptions).strip()
	description = remove_newlines(description)
	return(description)





def concatenate_with_bar_delim(*tokens):
	"""
	Concatenates any number of passed in tokens with a bar character and returns the 
	resulting string. This is useful for preparing things like gene names for entry
	into a table that could be written to a csv where multiple elements need to go 
	into a single column but a standard delimiter like a comma should not be used.
	Args:
	    *tokens: Description
	Returns:
	    TYPE: Description
	"""
	tokens = [token.split("|") for token in tokens]
	tokens = itertools.chain.from_iterable(tokens)
	tokens = filter(None, tokens)
	joined = "|".join(tokens).strip()
	joined = remove_newlines(joined)
	return(joined)







def append_words(description, words):
	"""
	Appends all words in a list of words to the end of a description string and returns
	the resulting larger string. This is useful in cases where it's desired to generate
	variance in word-choice but the structure of the description itself is not important
	and can be ignored, such as in using bag-of-words or similar technique.
	Args:
	    description (str): Any string, a description of something.
	    words (list): Strings to be appended to the description.
	Returns:
	    str: The description with new words appended.
	"""
	combined_list = [description]
	combined_list.extend(words)
	description = " ".join(combined_list).strip()
	return(description)





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












































