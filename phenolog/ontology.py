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
import os
import sys
import glob


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
	    (dict,dict): The forward and reverse mapping dictionaries.
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






def annotate_with_rabin_karp(object_dict, term_dict):
	"""Build a dictionary of annotations using Rabin Karp search.
	Args:
	    object_dict (dict): Mapping from IDs to natural language descriptions.
	    term_dict (dict): Mapping from strings to ontology term IDs.
	
	Returns:
	    dict: Mapping from object (phenotype) IDs to ontology term IDs.
	"""
	annotations = defaultdict(list)
	prime = 101
	for identifer,description in object_dict.items():
		for word,term_list in term_dict.items():
			if binary_search_rabin_karp(word, description, prime):
				annotations[identifer].extend(term_list)
	return(annotations)







def annotate_with_noble_coder(object_dict, path_to_jarfile, ontology_names, precise=1):
	"""Build a dictionary of annotations using external tool NOBLE Coder.
	Args:
	    object_dict (dict): Mapping from object IDs to natural language descriptions.
	    path_to_jarfile (str): Path to the jar file for the NOBLE Coder tool.
	    ontology_names (list): Strings used to find the correct terminology file, should match obo files too.
	    precise (int, optional): Set to 1 to do precise matching, set to 0 to accept partial matches.
	Returns:
	    dict: Mapping from object (phenotype) IDs to ontology term IDs.
	Raises:
	    FileNotFoundError: NOBLE Coder will check for a terminology file matching this ontology.
	"""

	# Configuration for running the NOBLE Coder script.
	tempfiles_directory = "temp_textfiles"
	output_directory = "temp_output"
	if not os.path.exists(tempfiles_directory):
		os.makedirs(tempfiles_directory)
	default_results_filename = "RESULTS.tsv"
	default_results_path = os.path.join(output_directory,default_results_filename)
	if precise == 1:
		specificity = "precise-match"
	else:
		specificity = "partial-match"


	# Generate temporary text files for each of the text descriptions.
	# Identifiers for descriptions are encoded into the filenames themselves.
	annotations = defaultdict(list)
	for identifier,description in object_dict.items():
		tempfile_path = os.path.join(tempfiles_directory, f"{identifier}.txt")
		with open(tempfile_path, "w") as file:
			file.write(description)


	# Use all specified ontologies to annotate each text file.
	# Also NOBLE Coder will check for a terminology file matching this ontology, check it's there.
	for ontology_name in ontology_names:
		expected_terminology_file = os.path.expanduser(os.path.join("~",".noble", "terminologies", f"{ontology_name}.term"))
		if not os.path.exists(expected_terminology_file):
			raise FileNotFoundError(expected_terminology_file)
		os.system(f"java -jar {path_to_jarfile} -terminology {ontology_name} -input {tempfiles_directory} -output {output_directory} -search '{specificity}' -score.concepts")
		default_results_filename = "RESULTS.tsv"		
		for identifier,term_list in _parse_noble_coder_results(default_results_path).items():
			annotations[identifier].extend(term_list)


	# Cleanup and return the annotation dictionary.
	_cleanup_noble_coder_results(output_directory, tempfiles_directory)
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



















def _parse_noble_coder_results(results_filename):
	"""
	Returns a mapping from object IDs to ontology term IDs inferred from reading
	a NOBLE Coder output file, this is helper function.
	Args:
	    results_filename (str): Path to the output file created by NOBLE Coder.
	Returns:
	    dict: Mapping from object (phenotype) IDs to ontology term IDs.
	"""
	df = pd.read_csv(results_filename, usecols=["Document", "Matched Term", "Code"], sep="\t")
	annotations = defaultdict(list)
	for row in df.itertuples():
		textfile_processed = row[1]
		identifer = str(textfile_processed.split(".")[0])
		tokens_matched = row[2].split()
		ontology_term_id = row[3]
		annotations[identifer].append(ontology_term_id)
	return(annotations)



def _cleanup_noble_coder_results(output_directory, textfiles_directory):
	"""
	Removes all directories and files created and used by running NOBLE Coder.
	This is a helper function.
	Args:
	    output_directory (str): Path to directory containing NOBLE Coder outputs.
	    textfiles_directory (str): Path to the directory of input text files.
	"""

	# Expected paths to each object that should be removed.
	html_file = os.path.join(output_directory,"index.html")
	results_file = os.path.join(output_directory,"RESULTS.tsv")
	properties_file = os.path.join(output_directory,"search.properties")
	reports_directory = os.path.join(output_directory,"reports")

	# Safely remove everything in the output directory.
	if os.path.isfile(html_file):
		os.remove(html_file)
	if os.path.isfile(results_file):
		os.remove(results_file)
	if os.path.isfile(properties_file):
		os.remove(properties_file)
	for filepath in glob.iglob(os.path.join(reports_directory,"*.html")):
		os.remove(filepath)
	os.rmdir(reports_directory)
	os.rmdir(output_directory)

	# Safely remove everything in the text file directory.
	for filepath in glob.iglob(os.path.join(textfiles_directory,"*.txt")):
		os.remove(filepath)
	os.rmdir(textfiles_directory)


















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








