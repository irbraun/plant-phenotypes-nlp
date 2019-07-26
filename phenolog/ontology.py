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




from phenolog.nlp import binary_search_rabin_karp







#  - Notes on problems working with PO and fastsemsim package - 
# There is some problem with how PO is treated by the fastsemsim utilities, not reproducible with GO or PATO.
# The [SemSim Object].util.lineage dictionary should map each node (string ID) in the ontology to it's root in
# the graph. It looks like the methods are set up so that there should only be one root (node without parents)
# for each node in the graph. I think those nodes are allowable in DAGs so I'm not sure why that is. But even 
# when editing the .obo ontology file add "is_a: Thing" edges to each term and creating the "Thing" root term 
# to provide a single root for the whole graph, the problem still persists and that dictionary only contains
# only a few terms. This is with version 1.0.0 of fastsemsim and po.obo file released on 6/5/2019.







def get_term_dictionaries(ontology_obo_file):
	"""
	Produces a mapping between ontology term IDs and a list of the strings which are related
	to them (the name of the term and any synonyms specified in the ontology) which is the
	forward dictionary, and a mapping between strings and all the ontology term IDs that those
	strings were associated with, which is the reverse mapping.
	Args:
	    ontology_obo_file (str): Path to the ontology file in the obo format.
	Returns:
	    tuple: The forward and reverse mapping dictionaries.
	"""
	forward_dict = {}
	reverse_dict = defaultdict(list)
	ontology = pronto.Ontology(ontology_obo_file)
	for term in ontology:
		if "obsolete" not in term.name:
			words = [term.name]
			words.extend([x.desc for x in list(term.synonyms)])
			forward_dict[term.id] = words
			for word in words:
				reverse_dict[word].append(term.id)
	return(forward_dict, reverse_dict)


def get_forward_term_dictionary(ontology_obo_file):
	"""Get one of the mapping types defined by the larger method.
	Args:
	    ontology_obo_file (TYPE): Description
	Returns:
	    TYPE: Description
	"""
	return(get_term_dictionaries(ontology_obo_file)[0])


def get_reverse_term_dictionary(ontology_obo_file):
	"""Get one of the mapping types defined by the larger method.
	Args:
	    ontology_obo_file (TYPE): Description
	Returns:
	    TYPE: Description
	"""
	return(get_term_dictionaries(ontology_obo_file)[1])





















def annotate_with_rabin_karp(object_dict, term_dict):
	"""Build a dictionary of annotations using Rabin Karp search.
	Args:
	    object_dict (dict): Mapping from IDs to natural language descriptions.
	    term_dict (dict): Mapping from strings to ontology term IDs.
	
	Returns:
	    dict: Mapping from IDs to ontology term IDs.
	"""
	#from nlp import rabin_karp_search
	annotations = defaultdict(list)
	prime = 101
	for identifer,description in object_dict.items():
		for word,term_list in term_dict.items():
			if binary_search_rabin_karp(word, description, prime):
				annotations[identifer].extend(term_list)
	return(annotations)






def write_annotations_to_tsv_file(annotations_dict, annotations_output_file):
	"""Create a tsv file of annotations that is compatable with fastsemsim.
	Args:
	    annotations_dict (dict): Mapping from IDs to lists of ontology term IDs.
	    annotation_output_file (str): Path to the output file to create. 
	"""
	outfile = open(annotations_output_file,'w')
	for identifer,term_list in annotations_dict.items():
		row_values = [str(identifer)]
		row_values.extend(term_list)
		outfile.write("\t".join(row_values).strip()+"\n")
	outfile.close()











def check_ontology(ontology_source_file):

	# Parameters for the ontology file.
	ontology_file_type = "obo"
	ontology_type = "Ontology"
	ignore_parameters = {}

	print("\n######################")
	print("# Loading ontology... #")
	print("######################\n")

	# Load the file.
	ontology = fss.load_ontology(source_file=ontology_source_file, ontology_type=ontology_type, file_type=ontology_file_type)

	print("\n#################################")
	print("# Ontology successfully loaded.")
	print("#################################\n")

	print("source_file: " + str(ontology_source_file))
	print("file_type: " + str(ontology_file_type))
	print("ontology_type: " + str(ontology_type))
	print("ignore_parameters: " + str(ignore_parameters))
	print("Number of nodes: " + str(ontology.node_number()))
	print("Number of edges: " + str(ontology.edge_number()))
	print("\nRoot terms in the ontology:\n-------------\n" + str(ontology.roots))
	print("\nType and number of edges:\n-------------\n" + str(ontology.edges['type'].value_counts()))
	print("-------------")
	print("\nInner edge number (within the ontology):\n-------------\n" + str(ontology.edges['inner'].value_counts()))
	print("-------------")
	print("\nIntra edge number (within the same namespace):\n-------------\n" + str(ontology.edges['intra'].value_counts()))
	print("-------------")
	print("\nOuter edges (link to other ontologies):\n-------------\n" + str(ontology.edges.loc[ontology.edges['inner'] == False]))
	print("-------------")
	print("\nInter edges (link between different namespaces - within the same ontology):\n-------------\n" + str(ontology.edges.loc[(ontology.edges['intra'] == False) & (ontology.edges['inner'] == True)]))
	print("-------------")






def check_annotations(ac_source_file):

	# Parameters for annotation corpus file with descriptions from fastsemsim documentation.
	ac_source_file_type = "plain"
	ac_params = {}
	ac_params['multiple'] = True 	# Set to True if there are many associations per line (the object in the first field is associated to all the objects in the other fields within the same line).
	ac_params['term first'] = False # Set to True if the first field of each row is a GO term. Set to False if the first field represents a protein/gene.
	ac_params['separator'] = "\t" 	# Select the separtor used to divide fields.

	print("\n#################################")
	print("# Loading annotation corpus.")
	print("#################################\n")

	# Load the file.
	ac = fss.load_ac(ontology, source_file=ac_source_file, file_type=ac_source_file_type, species=None, ac_descriptor=None, params = ac_params)

	print("\n#################################")
	print("# Annotation corpus successfully loaded.")
	print("#################################\n")

	print("\n\n")
	print("AC source: " + str(ac_source_file))
	print("ac source_type: " + str(ac_source_file_type))
	print("ac_parameters: " + str(ac_params))
	print("ac - Number of annotated proteins: " + str(len(ac.annotations)))
	print("ac - Number of annotated terms: " + str(len(ac.reverse_annotations)))
	print("The set of objects is: ", ac.obj_set)
	print("The set of terms is: ", ac.term_set)
	print("-------------")








