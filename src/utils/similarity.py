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









def get_similarity_df_using_ontologies(ontology_obo_file, annotated_corpus_tsv_file, object_dict):
	"""
	Method for creating a pandas dataframe of similarity values between all passed in object IDs
	based on the annotations of ontology terms to to all the natural language descriptions that
	those object IDs represent. The ontology files used have to be in the obo format currently and
	any terms that are annotated to the natural language (included in the annnotated corpus file)
	but are not found in the ontology are ignored when calculating similarity. This method also
	assumes that there is a single root to the DAG specified in the obo file which has ID "Thing".

	Args:
	    ontology_obo_file (str): File specifying the ontology that was used during annotation.
	    annotated_corpus_tsv_file (str): File specifying the annotations that were made.
	    object_dict (dict): Mapping between object IDs and the natural language descriptions. 
	
	Returns:
	    pandas.DataFrame: Each row in the dataframe is [first ID, second ID, similarity].
	"""

	# Intended values for loading the ontology from a generic obo file. Load the ontology.
	ontology_file_type = "obo"
	ontology_type = "Ontology"
	ignore_parameters = {}
	ontology = fss.load_ontology(source_file=ontology_obo_file, ontology_type=ontology_type, file_type=ontology_file_type)

	# Parameters for annotation corpus file with descriptions from fastsemsim documentation.
	ac_source_file_type = "plain"
	ac_params = {}
	ac_params['multiple'] = True 	# Set to True if there are many associations per line (the object in the first field is associated to all the objects in the other fields within the same line).
	ac_params['term first'] = False # Set to True if the first field of each row is a term. Set to False if the first field represents an object.
	ac_params['separator'] = "\t" 	# Select the separtor used to divide fields.
	ac = fss.load_ac(ontology, source_file=annotated_corpus_tsv_file, file_type=ac_source_file_type, species=None, ac_descriptor=None, params = ac_params)

	# Create the object for calculating semantic similarity.
	semsim_type='obj'
	semsim_measure='Jaccard'
	mixing_strategy='max'
	ss_util=None
	semsim_do_log=False
	semsim_params={}
	ss = fss.init_semsim(ontology=ontology, ac=ac, semsim_type=semsim_type, semsim_measure=semsim_measure, mixing_strategy=mixing_strategy, ss_util=ss_util, do_log=semsim_do_log, params=semsim_params)
	

	# Fix issue with lineage object in fastsemsim methods.
	ss.util.lineage = {}
	for node in ontology.nodes:
		ss.util.lineage[node] = "Thing"
	

	# Creating the batch semantic similarity object.
	ssbatch = fss.init_batchsemsim(ontology = ontology, ac = ac, semsim=ss)


	# Generate the pairwise calculations (dataframe) for this batch.
	object_list = list(object_dict.keys())
	result = ssbatch.SemSim(query=object_list, query_type="pairwise")
	result.columns = ["from", "to", "similarity"]
	return(result)










def get_similarity_df_using_doc2vec(doc2vec_model_file, object_dict):
	"""
	Method for creating a pandas dataframe of similarity values between all passed in object IDs
	using vector embeddings inferred for each natural language description using the passed in 
	Doc2Vec model, which could have been newly trained on relevant data or taken as a pretrained
	model. No assumptions are made about the format of the natural language descriptions, so any
	preprocessing or cleaning of the text should be done prior to being provied in the dictionary
	here.
	
	Args:
	    doc2vec_model_file (str): File where the Doc2Vec model to be loaded is stored.
	    object_dict (dict): Mapping between object IDs and the natural language descriptions. 
	
	Returns:
	    pandas.DataFrame: Each row in the dataframe is [first ID, second ID, similarity].
	"""


	model = gensim.models.Doc2Vec.load(doc2vec_model_file)

	vectors = []
	identifier_to_index_in_matrix = {}
	for identifier,description in object_dict.items():
		inferred_vector = model.infer_vector(description.lower().split())
		index_in_matrix = len(vectors)
		vectors.append(inferred_vector)
		identifier_to_index_in_matrix[identifier] = index_in_matrix

	matrix = cosine_similarity(vectors)

	result = pd.DataFrame(columns=["from", "to", "similarity"])
	for (p1, p2) in list(itertools.combinations_with_replacement(object_dict, 2)):	
		row = [p1, p2, matrix[identifier_to_index_in_matrix[p1]][identifier_to_index_in_matrix[p2]]]
		result.loc[len(result)] = row

	return(result)













# https://towardsdatascience.com/overview-of-text-similarity-metrics-3397c4601f50
def get_cosine_sim_matrix(*strs):
	vectors = [t for t in get_count_vectors(*strs)]
	similarity_matrix = cosine_similarity(vectors)
	return similarity_matrix

def get_count_vectors(*strs):
	text = [t for t in strs]
	vectorizer = CountVectorizer(text)
	vectorizer.fit(text)
	return vectorizer.transform(text).toarray()



def get_jaccard_sim_matrix(*strs):
	vectors = [t for t in get_binary_vectors(*strs)]
	dist = DistanceMetric.get_metric("jaccard")
	similarity_matrix = dist.pairwise(vectors)
	similarity_matrix = 1 - similarity_matrix
	return similarity_matrix

def get_binary_vectors(*strs):
	text = [t for t in strs]
	vectorizer = HashingVectorizer(text)
	vectorizer.fit(text)
	return vectorizer.transform(text).toarray()










def get_similarity_df_using_bagofwords(object_dict):
	"""
	Method for creating a pandas dataframe of similarity values between all passed in object IDs
	using vectors to represent each of the natural language descriptions as a bag-of-words. No 
	assumptions are made about the format of the natural language descriptions, so any cleaning
	or preprocessing of the text should be done prior to being provied in the dictionary here.
	
	Args:
	    object_dict (dict): Mapping between object IDs and the natural language descriptions. 
	
	Returns:
	    pandas.DataFrame: Each row in the dataframe is [first ID, second ID, similarity].
	"""


	descriptions = []
	identifier_to_index_in_matrix = {}
	for identifier,description in object_dict.items():
		index_in_matrix = len(descriptions)
		descriptions.append(description)
		identifier_to_index_in_matrix[identifier] = index_in_matrix

	matrix = get_cosine_sim_matrix(*descriptions)

	result = pd.DataFrame(columns=["from", "to", "similarity"])
	for (p1, p2) in list(itertools.combinations_with_replacement(object_dict, 2)):	
		row = [p1, p2, matrix[identifier_to_index_in_matrix[p1]][identifier_to_index_in_matrix[p2]]]
		result.loc[len(result)] = row

	return(result)










def get_similarity_df_using_setofwords(object_dict):
	"""
	Method for creating a pandas dataframe of similarity values between all passed in object IDs
	using vectors to represent each of the natural language descriptions as a set-of-words. No 
	assumptions are made about the format of the natural language descriptions, so any cleaning
	or preprocessing of the text should be done prior to being provied in the dictionary here.
	
	Args:
	    object_dict (dict): Mapping between object IDs and the natural language descriptions. 
	
	Returns:
	    pandas.DataFrame: Each row in the dataframe is [first ID, second ID, similarity].
	"""

	descriptions = []
	identifier_to_index_in_matrix = {}
	for identifier,description in object_dict.items():
		index_in_matrix = len(descriptions)
		descriptions.append(description)
		identifier_to_index_in_matrix[identifier] = index_in_matrix

	matrix = get_jaccard_sim_matrix(*descriptions)

	result = pd.DataFrame(columns=["from", "to", "similarity"])
	
	for (p1, p2) in list(itertools.combinations_with_replacement(object_dict, 2)):	
		row = [p1, p2, matrix[identifier_to_index_in_matrix[p1]][identifier_to_index_in_matrix[p2]]]
		result.loc[len(result)] = row

	return(result)






