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








def get_wordnet_related_words(word, context):
	"""
	Method to generate a list of words that are found to be related to the input word through
	the WordNet ontology/resource. The correct sense of the input word to used within the
	context of WordNet is picked based on disambiguation from PyWSD package which is taking
	the surrounding text (or whatever text is provided as context) into account. All synonyms,
	hypernyms, and hyponyms are considered to be related words in this case.
	
	Args:
	    word (str): The word for which we want to find related words.
	    context (str): Text to use for word-sense disambigutation, usually sentence the word is in.
	
	Returns:
	    list: The list of related words that were found, could be empty if nothing was found.
	"""


	# To get the list of synsets for this word if not using disambiguation.
	list_of_possible_s = wordnet.synsets(word)

	# Disambiguation of synsets (https://github.com/alvations/pywsd).
	# Requires installation of non-conda package PyWSD from pip ("pip install pywsd").
	# The methods of disambiguation that are supported by this package are: 
	# (simple_lesk, original_lesk, adapted_lesk, cosine_lesk, and others). 
	from pywsd.lesk import cosine_lesk
	s = cosine_lesk(context, word)

	try:
		# Generate related words using wordnet, including synonyms, hypernyms, and hyponyms.
		# The lists of hypernyms and hyponyms need to be flattened because they're lists of lists from synsets.
		# definition() yields a string. 
		# lemma_names() yields a list of strings.
		# hypernyms() yields a list of synsets.
		# hyponyms() yields a list of synsets.
		synset_definition = s.definition()
		synonym_lemmas = s.lemma_names() 													
		hypernym_lemmas_nested_list = [x.lemma_names() for x in s.hypernyms()] 
		hyponym_lemmas_nested_list = [x.lemma_names() for x in s.hyponyms()]
		# Flatten those lists of lists.
		hypernym_lemmas = [word for lemma_list in hypernym_lemmas_nested_list for word in lemma_list]
		hyponym_lemmas = [word for lemma_list in hyponym_lemmas_nested_list for word in lemma_list]

		# Print out information about the synset that was picked during disambiguation.
		#print(synset_definition)
		#print(synonym_lemmas)
		#print(hypernym_lemmas)
		#print(hyponym_lemmas)
		
		return(synonym_lemmas+hypernym_lemmas+hyponym_lemmas)
	except AttributeError:
		return([])







def get_word2vec_related_words(word, model, threshold, max_qty):
	"""
	Method to generate a list of words that are found to be related to the input word through
	assessing similarity to other words in a word2vec model of word embeddings. The model can
	be learned from relevant text data or can be pretrained on an existing source. All words
	that satisfy the threshold provided up to the quantity specified as the maximum are added.

	Args:
	    word (str): The word for which we want to find other related words.
	    model (Word2Vec): The actual model object that has already been loaded.
	    threshold (float): Similarity threshold that must be satisfied to add a word as related.
	    max_qty (int): Maximum number of related words to accept.
	
	Returns:
	    list: The list of related words that were found, could be empty if nothing was found.
	"""
	
	related_words = []
	matches = model.most_similar(word, topn=max_qty)
	for match in matches:
		word_in_model = match[0]
		similarity = match[1]
		if (similarity >= threshold):
			related_words.append(word_in_model)
	return(related_words)









# Methods to apply the above methods to an entire description and return the flattened list of words.

def get_all_wordnet_related_words(description):
	flatten = lambda l: [item for sublist in l for item in sublist]
	return flatten([get_wordnet_related_words(word,description) for word in get_clean_token_list(description)])


def get_all_word2vec_related_words(description, model, threshold, max_qty):
	flatten = lambda l: [item for sublist in l for item in sublist]
	return flatten([get_word2vec_related_words(word, model, threshold, max_qty) for word in get_clean_token_list(description)])











def get_clean_description(description):
	translator = str.maketrans('', '', string.punctuation)
	description = description.translate(translator)
	description = description.lower()
	return(description)

def get_clean_token_list(description):
	translator = str.maketrans('', '', string.punctuation)
	description = description.translate(translator)
	token_list = description.lower().split()
	return(token_list)

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






