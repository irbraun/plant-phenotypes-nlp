import streamlit as st
import pandas as pd
import numpy as np
import sys
import re
import itertools
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import defaultdict
from string import punctuation
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import strip_non_alphanum, stem_text, preprocess_string, strip_tags, strip_punctuation


sys.path.append("../../oats")
import oats
from oats.utils.utils import load_from_pickle
from oats.biology.dataset import Dataset
from oats.annotation.ontology import Ontology



nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)





# API and good tutorials for using the streamlit package.
# https://docs.streamlit.io/en/stable/api.html
# https://towardsdatascience.com/how-to-write-web-apps-using-simple-python-for-data-scientists-a227a1a01582
# Whenever a widget value is changed, the whole script runs from top to bottom.
# Use caching in order to create functions that only run once. When the script reruns, streamlit
# checks the the arguments for functions with the cache decorator and if they have not changed
# then the function is not called.

# Good resource what is possible with the streamlit package.
# http://awesome-streamlit.org/
# Use the 'Table Experiments' app in the gallery to see the different types of ways that dataframes can be
# presented in a streamlit app. Using st.dataframe allows for showing a truncated dataframe with small columns
# that can be sorted on the fly in the app itself. The st.table class is not sortable or interactive automaticaly
# in the app without coding that in the app script, but it doesn't truncate anything. So this is better for
# displaying columns that have a lot of text in them when you don't to remove any of the text.


# TODO 
# Get running with other pickles including computational annotations and BERT.
# Make sure the application can be loaded with any other set of files and still works with these pickles.
# Make a separate notebook that just creates the pickles needed to build the application?
# Load this with the SNPedia dataset and see what happens.









# Markdown for introducing the app and linking to other relevant resources like the project Github page.

'''
# QuOATS
A tool for querying genes and phenotypes with ontology annoations and text similarity.

## Instructions
Use the searchbar below to enter a query. Select whether the query refers to a gene identifier, ontology terms, keywords and 
keyphrases, or free text. Examples of possible queries for each of these options are given. Gene identifiers can be gene names, 
gene models, protein names, or any other alias or synonym present in the dataset. Ontology terms should be separated by spaces 
or commas. Keywords and keyphrases should be separated by commas. There are no length or formatting restrictions for a free text 
query. If a free text query is performed, the selected similarity algorithm is used to quantify similarity to the
phenotype descriptions of all genes in the dataset.

Note: The tables on this page are best formatted when viewing in wide mode (Select '≡', then 'Settings', then 'Show app in wide mode)'.
'''








# Note that the output object(s) and dictionaries should not actually be modified in this script.
# But streamlit was detecting a change in the outputs of this function? So the allow output mutation
# decorator is necessary in order for this cached function to not be run again on refresh.
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def read_in_files(dataset_path, approach_names_and_data, approach_mapping_files):
	"""Read in the large files that initialize the application here, which takes a long time.
	
	Args:
		dataset_path (TYPE): Description
		approach_names_and_data (TYPE): Description
		approach_mapping_files (TYPE): Description
	
	Returns:
		TYPE: Description
	"""
	
	# Read in the dataset and create its object.
	dataset = Dataset(data=dataset_path, keep_ids=True)
	dataset.filter_has_description()

	# Reading in the large files that store objects for vectorizing and comparing texts.
	approach_to_object = {}
	for k,v in approach_names_and_data.items():
		obj = load_from_pickle(v["path_dists"])
		vectors = load_from_pickle(v["path_vectors"])
		obj.vector_dictionary = vectors
		approach_to_object[k] = obj

	#approach_to_object = {k:load_from_pickle(v["path_dists"]) for k,v in approach_names_and_data.items()}
	#for k,obj in approach_to_object.items():
	#	approach_to_object[k].vector_dictionary = load_from_approach_names_and_data[k]["path_vectors"]

	mapping_key_to_mapping_dict = {k:load_from_pickle(v) for k,v in approach_mapping_files.items()}
	approach_to_mapping = {k:mapping_key_to_mapping_dict[v["mapping"]] for k,v in approach_names_and_data.items()}

	# Other expensive computation here so that it's done inside the cached function.
	# Get the dataframe for this dataset and add some additional columns that will be useful for displaying information.
	# Slow additions to the dataframe should go here. Very fast ones can go in the main non-cached part of the script.
	df = dataset.to_pandas()
	df["Gene"] = df["id"].map(lambda x: dataset.get_gene_dictionary()[x].primary_identifier)
	df["Gene Model"] = df["id"].map(lambda x: dataset.get_gene_dictionary()[x].gene_models[0] if len(dataset.get_gene_dictionary()[x].gene_models)>0 else "")
	df["truncated_descriptions"] = df["descriptions"].map(lambda x: truncate_string(x, 80))
	id_to_descriptions_for_keyword_matching = {i:PREPROCESSING_FOR_KEYWORD_SEARCH_FUNCTION(s) for i,s in dataset.get_description_dictionary().items()}
	return(dataset, approach_to_object, approach_to_mapping, df, id_to_descriptions_for_keyword_matching)






@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def read_in_ontologies(names, paths):
	"""Read in ontology objects from .obo files and return them, can take a long time.
	
	Args:
		names (TYPE): Description
		paths (TYPE): Description
	
	Returns:
		TYPE: Description
	"""
	ontologies = {}
	for name,path, in zip(names,paths):
		#ontologies[name]=Ontology(path)
		ontologies[name] = load_from_pickle(path)
	return(ontologies)















def truncate_string(text, char_limit):
	"""Helpful formatting function for truncating strings to a certain maximum length.
	
	Args:
		text (TYPE): Description
		char_limit (TYPE): Description
	
	Returns:
		TYPE: Description
	"""
	truncated_text = text[:char_limit]
	if len(text)>char_limit:
		# Make extra room for the "..." string and then add it.
		truncated_text = text[:char_limit-3]
		truncated_text = "{}...".format(truncated_text)
	return(truncated_text)






def distance_float_to_similarity_int(distance):
	"""Generates a friendlier (0 to 100) similarity value for displaying in search results.
	
	Args:
		distance (TYPE): Description
	
	Returns:
		TYPE: Description
	"""
	similarity_float = 1-distance
	similarity_int = int(similarity_float*100)
	return(similarity_int)










def gene_name_search(dataset, gene_name):
	"""Helper function for searching the dataset for a gene identifier.
	
	Args:
		dataset (TYPE): Description
		gene_name (TYPE): Description
	
	Returns:
		TYPE: Description
	"""
	gene_name = gene_name.lower().strip()
	species_to_gene_id_list = defaultdict(list)
	for species in dataset.get_species():
		gene_ids = dataset.get_species_to_name_to_ids_dictionary(include_synonyms=True, lowercase=True)[species][gene_name]
		for gene_id in gene_ids:
			species_to_gene_id_list[species].append(gene_id)
	return(species_to_gene_id_list)



	

def keyword_search(id_to_text, raw_keywords, modified_keywords):
	"""Helper function for searching the dataset for keywords and keyphrases.
	
	Args:
		id_to_text (TYPE): Description
		raw_keywords (TYPE): Description
		modified_keywords (TYPE): Description
	
	Returns:
		TYPE: Description
	"""
	# The raw keywords and modified keywords should be two paired lists where the elements correspond to one another.
	# The modifications done to the keywords should already match the modifications done to the texts in the input dictionary so they can be directly compared.
	assert len(raw_keywords) == len(modified_keywords)
	id_to_found_keywords = {i:[r_kw for r_kw,m_kw in zip(raw_keywords,modified_keywords) if m_kw in text] for i,text in id_to_text.items()}
	id_to_num_found_keywords = {i:len(kw_list) for i,kw_list in id_to_found_keywords.items()}
	return(id_to_found_keywords, id_to_num_found_keywords)









def ontology_term_search(id_to_direct_annotations, id_to_indirect_annotations, term_ids, result_column_width):
	"""Helper function for searching the dataset for ontology term annotations.
	
	Args:
		id_to_direct_annotations (TYPE): Description
		id_to_indirect_annotations (TYPE): Description
		term_ids (TYPE): Description
		result_column_width (TYPE): Description
	
	Returns:
		TYPE: Description
	"""
	assert len(id_to_direct_annotations) == len(id_to_indirect_annotations)
	gene_id_to_direct_match = {i:[(term_id in direct_annotations) for term_id in term_ids] for i,direct_annotations in id_to_direct_annotations.items()}
	gene_id_to_indirect_match = {i:[(term_id in indirect_annotations) for term_id in term_ids] for i,indirect_annotations in id_to_indirect_annotations.items()}
	
	# Map each gene ID to a result string that says which terms matched and how.
	gene_id_to_result_string = {}
	gene_id_to_num_direct_matches = {}
	gene_id_to_num_indirect_matches = {}
	for gene_id in gene_id_to_direct_match.keys():		
		gene_id_to_num_direct_matches[gene_id] = sum(gene_id_to_direct_match[gene_id])
		gene_id_to_num_indirect_matches[gene_id] =  any(gene_id_to_indirect_match[gene_id])
		lines = []
		for idx,term_id in enumerate(term_ids):
			if gene_id_to_direct_match[gene_id][idx]:
				match_type = "Direct Annotation"
			elif gene_id_to_indirect_match[gene_id][idx]:
				match_type = "Inherited Annotation"
			else:
				continue
			# If there was a valid match found, create this formatted string.
			match_str = "({})".format(match_type)
			term_id_str = term_id
			num_chars_left_to_fill = result_column_width-len(term_id_str)-len(match_str)
			filler_str = "."*num_chars_left_to_fill
			line = "{}{}{}\n\n".format(term_id_str, filler_str, match_str)
			lines.append(line)
		result_str = "".join(lines)
		gene_id_to_result_string[gene_id] = result_str

	return(gene_id_to_num_direct_matches, gene_id_to_num_indirect_matches, gene_id_to_result_string)











def description_search(text, graph, tokenization_function, preprocessing_function, result_column_width, result_column_max_lines):
	"""Helper function for searching the dataset for similar phenotype descriptions.
	
	Args:
		text (TYPE): Description
		graph (TYPE): Description
		tokenization_function (TYPE): Description
		preprocessing_function (TYPE): Description
		result_column_width (TYPE): Description
		result_column_max_lines (TYPE): Description
	
	Returns:
		TYPE: Description
	"""
	# Do tokenization and preprocessing on the searched text to yield a list of strings.
	# The tokenization and preprocessing have to match the object that will be used, and this decision is informed by
	# some mapping that is already done, and the choices are passed in to this function.
	# The tokenization might just yield a single text, but it still goes in a list for consistency.
	sentence_tokens = tokenization_function(text)
	preprocessed_sentence_tokens  = [preprocessing_function(s) for s in sentence_tokens]


	# Get a mapping between gene IDs and their distances to each new text string parsed from the search string.
	gene_id_to_distances = defaultdict(list)
	for s in preprocessed_sentence_tokens:
		internal_id_to_distance_from_new_text = graph.get_distances(s)
		for gene_id,graph_ids in gene_id_to_graph_ids.items():
			# What's the smallest distances between this new graph and one of the texts for the internal graph nodes.
			graph_distances = [internal_id_to_distance_from_new_text[graph_id] for graph_id in graph_ids]
			min_graph_distance = min(graph_distances)
			# Remember that value as the distance from this one parsed text string from the search to this particular gene.
			gene_id_to_distances[gene_id].append(min_graph_distance)


	# Making a highly formatted string to show how distances break down by individual sentences tokens.
	# This is very specific to how the column for that information is presented, change how it looks here.
	# Some of this formatting is a work-around for not having alot of control over column widths and text-wrapping in the streamlit table.
	gene_id_to_result_string = {}
	for gene_id, distances in gene_id_to_distances.items():
		lines_with_dist_list = []
		for s,d in zip(sentence_tokens,distances):
			parsed_string_truncated = truncate_string(s, result_column_width-5)
			similarity_string = "({})".format(distance_float_to_similarity_int(d))
			num_chars_left_to_fill = result_column_width-len(parsed_string_truncated)-len(similarity_string)
			parsed_string_truncated = parsed_string_truncated + "."*num_chars_left_to_fill
			#line = "{}({:.2f})\n\n".format(parsed_string_truncated, d)

			#newline_string = "<br>"
			newline_string = r"\n\n"

			line = "{}{}{}".format(parsed_string_truncated, similarity_string, newline_string)
			lines_with_dist_list.append((line,d))
		# Sort that list of lines by distance, because we only want to show the best matches if the description is very long.
		lines_with_dist_list = sorted(lines_with_dist_list, key=lambda x: x[1])
		line_string = "".join([x[0] for x in lines_with_dist_list][:result_column_max_lines])
		gene_id_to_result_string[gene_id] = line_string




	# For now, just get a mapping between gene IDs and their minimum distance to any of those parsed strings.
	# Maybe this should take more than just the minimum of them into account for this application?
	gene_id_to_min_distance = {}
	for gene_id,distances in gene_id_to_distances.items():
		gene_id_to_min_distance[gene_id] = min(distances)

	return(gene_id_to_result_string, gene_id_to_min_distance)

























# Dangerous part. These have to exactly match how the text is treated within the notebook that generates the pairwise distances.
# There's no explicit check in the code that makes sure the processing is identical between those two locations.
# In the future this should probably be changed so both sets of code are calling one resources that knows how to do these things.
sentence_tokenize = lambda text: sent_tokenize(text)
as_one_token = lambda text: [text]
identify_function = lambda text: text
simple_preprocessing = lambda text: " ".join(simple_preprocess(text))
full_preprocessing = lambda text: " ".join(preprocess_string(text))





# Where are the dataset and ontology files that should be loaded for this application?
DATASET_PATH = "resources/genes_texts_annots.csv"
ONTOLOGY_NAMES = ["PATO","PO","GO"]
ONTOLOGY_OBO_PATHS = ["resources/pato.obo", "resources/po.obo", "resources/go.obo"]
ONTOLOGY_PICKLE_PATHS = ["resources/pato.pickle", "resources/po.pickle", "resources/go.pickle"]




# What are the different fields for each approach in this nested dictionary?
# path: The path to the pickle for loading the oats object associated with this approach.
# mapping: A key that allows for retrieving the correct mapping between IDs used here and those used internally by each approach.
# tokenization_function: A function for how text should be tokenized in order to be compatible with this approach.
# preprocessing_function: A function for how text should be preprocessed in order to be compatible with this approach.
APPROACH_NAMES_AND_DATA = {
	# "N-Grams (Whole Phenotype)":{
	# 	"path":"resources/dists_with_n_grams_full_words_1_grams_tfidf.pickle", 
	# 	"mapping":"whole_texts",
	# 	"tokenization_function":as_one_token,
	# 	"preprocessing_fucntion":full_preprocessing,
	# 	},
	# "N-Grams (Sentence Tokenized)":{
	# 	"path":"resources/dists_with_n_grams_tokenization_full_words_1_grams_tfidf.pickle", 
	# 	"mapping":"sent_tokens",
	# 	"tokenization_function":sentence_tokenize,
	# 	"preprocessing_fucntion":full_preprocessing,
	# 	},
	# "word2vec-tokenized":{
	# 	"path":"resources/wtok.pickle", 
	# 	"mapping":"sent_tokens",
	# 	"tokenization_function":sentence_tokenize,
	# 	"preprocessing_fucntion":identify_function,
	# 	},
	# "doc2vec":{
	#  	"path":"resources/d.pickle", 
	#  	"mapping":"whole_texts",
	#  	"tokenization_function":as_one_token,
	#  	"preprocessing_fucntion":identify_function,
	#  	},


	"n-grams":{
		"path_dists":"resources/dists_with_n_grams_tokenization_full_words_1_grams_tfidf.pickle", 
		"path_vectors":"resources/vectors_with_n_grams_tokenization_full_words_1_grams_tfidf.pickle",
		"mapping":"sent_tokens_N",
		"tokenization_function":sentence_tokenize,
		"preprocessing_fucntion":full_preprocessing,
		},
	"fuzzy":{
		"path_dists":"resources/dists_with_word2vec_tokenization_pubmed_size_200_mean.pickle", 
		"path_vectors":"resources/vectors_with_word2vec_tokenization_pubmed_size_200_mean.pickle",
		"mapping":"sent_tokens_W",
		"tokenization_function":sentence_tokenize,
		"preprocessing_fucntion":identify_function,
		},


	}


# For testing, be able to subset this nested dictionary without having to uncomment sections of it.
# Just uncomment these two lines to use the entire set of approaches and load all files.
names_to_actually_use = ["fuzzy", "n-grams"]
APPROACH_NAMES_AND_DATA = {k:v for k,v in APPROACH_NAMES_AND_DATA.items() if k in names_to_actually_use}




# Mappings between key strings and the files that should be used to load dictionaries for converting between IDs.
# The keys for this dictionary should map the items from the nested dictionay above, so that these mappings from 
# these dictinoary files can then be associated with each of the approaches from the nested dictionary.
APPROACH_MAPPING_FILES = {
	"whole_texts":"resources/gene_id_to_unique_ids_whole_texts.pickle",
	"sent_tokens_N":"resources/N_gene_id_to_unique_ids_sent_tokens.pickle",
	"sent_tokens_W":"resources/W_gene_id_to_unique_ids_sent_tokens.pickle",



	"go_term_sets":"resources/gene_id_to_unique_ids_go_term_sets.pickle",
	"po_term_sets":"resources/gene_id_to_unique_ids_po_term_sets.pickle",
	"go_terms":"resources/gene_id_to_unique_ids_go_terms.pickle",
	"po_terms":"resources/gene_id_to_unique_ids_po_terms.pickle",
	}





# How should keywords and phrases be cleaned and handled as far as preprocessing or stemming goes?
KEYWORD_DELIM = "[DELIM]"
KEYWORD_PREPROCESSING_FILTERS = [lambda x: x.lower(), strip_non_alphanum, strip_tags, strip_punctuation, stem_text]
PREPROCESSING_FOR_KEYWORD_SEARCH_FUNCTION = lambda x: "{}{}{}".format(KEYWORD_DELIM, KEYWORD_DELIM.join([token for token in preprocess_string(x, KEYWORD_PREPROCESSING_FILTERS)]), KEYWORD_DELIM)



# Some options for how aspects of the tables that are presented after each search look.
RESULT_COLUMN_STRING = "___________________________________Matches"
RESULT_COLUMN_STRING = "                                   Matches"
MAX_LINES_IN_RESULT_COLUMN = 10

















# The first dictionary created here maps the name of an approach, like "n-grams" to an oats.distances object.
# The second dictionary created here maps the name of an approach like "n-grams" to another dictionary that
# maps the IDs that in this dataset to the IDs that are used internally by that corresponding distances object.
# This is necessary because those internal IDs represent compressed or altered versions of the text and could 
# refer to more than one text instance in the actual dataset that is seen here.
with st.spinner("Reading very large files, this might take a few minutes."):
	dataset, approach_to_object, approach_to_mapping, df, id_to_descriptions_for_keyword_matching= read_in_files(DATASET_PATH, APPROACH_NAMES_AND_DATA, APPROACH_MAPPING_FILES)
	ontologies = read_in_ontologies(ONTOLOGY_NAMES, ONTOLOGY_PICKLE_PATHS)



# These mappings are necessary if the internal strings used for species and how they should be displayed are different.
internal_species_strings = ["ath", "zma", "sly", "gmx", "osa", "mtr"]
display_species_strings = ["Arabidopsis", "Maize", "Tomato", "Soybean", "Rice", "Medicago"]
to_species_display_name = {i:d for i,d in zip(internal_species_strings,display_species_strings)}











############# The Sidebar ###############


# Presenting the options for filtering by species.
st.sidebar.markdown("### Filtering by Species")
species_display_names = [to_species_display_name[x] for x in dataset.get_species()]
species_list = st.sidebar.multiselect(label="Filter to only include certain species", options=species_display_names)
if len(species_list) == 0:
	species_list = species_display_names

# Present information and search options for finding a particular gene or protein in the sidebar.
st.sidebar.markdown("### Select a Similarity Algorithm")
approach = st.sidebar.selectbox(label="Select one", options=list(approach_to_object.keys()), index=0)

# Variables that get set according to which approach was selected.
graph = approach_to_object[approach]
gene_id_to_graph_ids = approach_to_mapping[approach]




# Presenting the general options for how the data is displayed.
st.sidebar.markdown("### General Page Options")
truncate = st.sidebar.checkbox(label="Truncate long phenotypes", value=True)
synonyms = st.sidebar.checkbox(label="Show possible gene synonyms", value=False)
include_examples = st.sidebar.checkbox(label="Include Examples", value=False)



table_width = st.sidebar.slider(label="Table Width", min_value=100, max_value=8000, value=1000, step=100, format=None, key=None)


# Presenting some more advanced options that shouldn't normally need to be changed.
#st.sidebar.markdown("### Advanced Options")











############# The Searchbar Section ###############




# Display the search section of the main page.
st.markdown("## Search")

search_types = ["gene", "ontology", "keyword", "phenotype"]
search_types_labels = ["Gene Identifiers", "Ontology Terms", "Keywords & Keyphrases", "Free Text"]
search_type_examples = ["epc1", "PATO:0000587, PATO:0000069", "dwarfism, root system, leaves", "Plants are reduced in height. Plants have wide leaves. Smaller than normal. Something else too."]

if include_examples:
	search_type_label_map = {t:"{} (e.g. '{}')".format(l,e) for t,l,e in zip(search_types, search_types_labels, search_type_examples)}
else:
	search_type_label_map = {t:l for t,l in zip(search_types,search_types_labels)}


search_types_format_func = lambda x: search_type_label_map[x]
search_type = st.radio(label="Select a type of search", options=search_types, index=0, format_func=search_types_format_func)
input_text = st.text_input(label="Enter text here")






















# TODO add download buttons.














############## Default view, presenting the entire dataset ################



# Subset the dataframe if necessary.

# The dataframe as it should be formatted for displaying in the app, not all data is needed.
#df = dataset.to_pandas()
#df["Species"] = df["species"].map(to_species_display_name)
df["Species"] = df["species"].map(to_species_display_name)
df = df[df["Species"].isin(species_list)]
#df["Gene"] = df["id"].map(lambda x: dataset.get_gene_dictionary()[x].primary_identifier)
#df["Gene Model"] = df["id"].map(lambda x: dataset.get_gene_dictionary()[x].gene_models[0] if len(dataset.get_gene_dictionary()[x].gene_models)>0 else "")
#df["Phenotype Description"] = df["descriptions"]
#df["Phenotype Description (Truncated)"] = df["descriptions"].map(lambda x: truncate_string(x, 800))



if truncate:
	df["Phenotype Description"] = df["truncated_descriptions"]
else:
	df["Phenotype Description"] = df["descriptions"]














############### Listening for something to typed and entered into any of the search boxes #############






################ Searching by GENE IDENTIFIER ################ 

if search_type == "gene" and input_text != "":

	# Start the results section of the page.
	st.markdown("## Results")
	
	# Do the actual processing of the search against the full dataset here.
	gene_search_string = input_text
	gene_matches = gene_name_search(dataset=dataset, gene_name=gene_search_string)

	# Search text was entered and the search was processed but the list of relevant IDs found is empty.
	if len(gene_matches)==0:
		st.markdown("No genes were found for '{}'.".format(gene_search_string))

	# Search text was entered and the search was processed and atleast one matching ID was found.
	gene_buttons_dict = {}
	if len(gene_matches)>0:
		st.markdown("Genes matching '{}' are shown below. Select one to see other genes with similarly described phenotypes.".format(gene_search_string))
		unique_button_key = 0
		for species,id_list in gene_matches.items():
			for i in id_list:
				primary_gene_name = dataset.get_gene_dictionary()[i].primary_identifier
				other_names = dataset.get_gene_dictionary()[i].all_identifiers
				button_label = "{}: {}".format(to_species_display_name[species], primary_gene_name)
				gene_buttons_dict[i] = st.button(label=button_label, key=unique_button_key)
				unique_button_key = unique_button_key+1
				if synonyms:
					synonyms_field_char_limit = 150
					synonyms_field_str = truncate_string(", ".join(other_names), synonyms_field_char_limit)
					st.markdown("(Other synonyms include {})".format(synonyms_field_str))



	# Handle what should happen if any of the previously presented gene buttons was clicked.
	# Has to be a loop because we need to check all the presented buttons, and there might be more than one.
	for i,gene_button in gene_buttons_dict.items():
		if gene_button:
			
			# Get information about which gene from the dataset was selected.
			selected_gene_primary_name = dataset.get_gene_dictionary()[i].primary_identifier
			selected_gene_other_names = dataset.get_gene_dictionary()[i].all_identifiers
			selected_gene_phenotype_description = dataset.get_description_dictionary()[i]
			
			# Describe what the results of the search are and what they mean in markdown.
			st.markdown("### Gene Information")
			st.markdown("**Identifier:** {}".format(selected_gene_primary_name))
			st.markdown("**Possible Synonym(s):** {}".format(", ".join(selected_gene_other_names)))
			st.markdown("**Phenotype Description(s):** {}".format(selected_gene_phenotype_description))
			st.markdown("### Genes with Similar Phenotypes")
			#st.markdown("The genes in the dataset below are sorted by similarity to **{}**, as determined by the selected phenotype similarity approach in the sidebar.".format(selected_gene_primary_name))

			
			# Perform a secondary search using this gene's phenotype description to organize the rest of the data.
			with st.spinner("Searching dataset for other genes with similar phenotype descriptions..."):
				search_string = dataset.get_description_dictionary()[i]
				f_tokenizing = APPROACH_NAMES_AND_DATA[approach]["tokenization_function"]
				f_preprocessing = APPROACH_NAMES_AND_DATA[approach]["preprocessing_fucntion"]
				gene_id_to_result_string, gene_id_to_min_distance =  description_search(search_string, graph, f_tokenizing, f_preprocessing, len(RESULT_COLUMN_STRING), MAX_LINES_IN_RESULT_COLUMN)
				df[RESULT_COLUMN_STRING] = df["id"].map(gene_id_to_result_string)
				df["distance"] = df["id"].map(gene_id_to_min_distance)
				df.sort_values(by=["distance","id"], ascending=[True,True], inplace=True)


			# Display the sorted and filtered dataset as a table with the relevant columns.
			df.index = np.arange(1, len(df)+1)
			st.table(data=df[[RESULT_COLUMN_STRING, "Species", "Gene", "Gene Model", "Phenotype Description"]])









################ Searching by ONTOLOGY TERMS ################ 

elif search_type == "ontology" and input_text != "":

	st.markdown("## Results")


	term_ids = input_text.replace(","," ").split()

	

	pattern = re.compile("[A-Z]{2,}[:_][0-9]{7}$")
	term_ids_are_valid = [bool(pattern.match(term_id)) for term_id in term_ids]
	if False in term_ids_are_valid:
		st.markdown("Invalid ontology term identifiers.")




		
	else:
		# Building the annotations dictionary. This should actually be done outside the search sections.
		direct_annotations = defaultdict(list)
		inherited_annotations = defaultdict(list)
		for ontology_name,ontology_obj in ontologies.items():
			annotations_with_this_ontology = dataset.get_annotations_dictionary(ontology_name)
			for i in dataset.get_ids():
				direct_annotations[i].extend(annotations_with_this_ontology[i])
				inherited_annotations[i].extend(ontology_obj.inherited(annotations_with_this_ontology[i]))



		# Linking out to other resources like Ontobee and Planteome.
		# TODO Verify that the links lead somewhere valid before displaying them in the application.
		ontobee_url_template = "http://www.ontobee.org/ontology/{}?iri=http://purl.obolibrary.org/obo/{}_{}"
		planteome_url_template = "http://browser.planteome.org/amigo/term/{}:{}"
		lines_with_terms_and_links = []
		for term_id in term_ids:
			term_id_str = term_id.replace(":","_")
			ontology_name, term_number = tuple(term_id_str.split("_"))
			try:
				term_label = ontologies[ontology_name.upper()][term_id].name
				ontobee_url = ontobee_url_template.format(ontology_name, ontology_name, term_number)
				planteome_url = planteome_url_template.format(ontology_name, term_number)
				line = "{} ({}, [Ontobee]({}), [Planteome]({}))".format(term_id, term_label, ontobee_url, planteome_url)
				lines_with_terms_and_links.append(line)
			except:
				pass
		lines_with_terms_and_links_str = "\n\n".join(lines_with_terms_and_links)

		st.markdown("### Ontology Term Information")
		st.markdown(lines_with_terms_and_links_str)


		gene_id_to_num_direct_matches, gene_id_to_num_indirect_matches, gene_id_to_result_string = ontology_term_search(direct_annotations, inherited_annotations, term_ids, len(RESULT_COLUMN_STRING))
		df[RESULT_COLUMN_STRING] = df["id"].map(gene_id_to_result_string)
		df["num_direct"] = df["id"].map(gene_id_to_num_direct_matches)
		df["num_indirect"] = df["id"].map(gene_id_to_num_indirect_matches)
		subset_df = df[(df["num_direct"]>0) | (df["num_indirect"]>0)]
		subset_df.sort_values(by=["num_direct","num_indirect","id"], ascending=[False,False,True], inplace=True)


		subset_df.index = np.arange(1, len(subset_df)+1)

		if subset_df.shape[0] == 0:
			st.markdown("No genes were found for '{}'. Make sure the ontology term IDs are separated by commas or spaces and formatted like the examples.".format(term_id))
		else:
			# Describe what the results of the search are and what they mean in markdown.
			st.markdown("### Genes with Matching Annotations")
			st.markdown("The dataset of plant genes below shows only genes with annotations that included one or more of the searched ontology terms.")

			# Display the sorted and filtered dataset as a table with the relevant columns.
			st.table(data=subset_df[[RESULT_COLUMN_STRING, "Species", "Gene", "Gene Model", "Phenotype Description"]])








################ Searching by KEYWORDS AND KEYPHRASES ################ 

elif search_type == "keyword" and input_text != "":

	# Start the results section of the page.
	st.markdown("## Results")

	# Do the processing of the search and add necessary columns to the dataframe.
	with st.spinner("Searching dataset for keywords or phrases..."):	
		search_kws = input_text
		keywords = search_kws.strip().strip(punctuation).split(",")
		raw_keywords = [kw.strip() for kw in keywords]
		modified_keywords = [PREPROCESSING_FOR_KEYWORD_SEARCH_FUNCTION(kw) for kw in raw_keywords]
		id_to_found_keywords, id_to_num_found_keywords = keyword_search(id_to_descriptions_for_keyword_matching, raw_keywords, modified_keywords)
		df["num_found"] = df["id"].map(id_to_num_found_keywords)
		subset_df = df[df["num_found"]>0]
		subset_df["Keywords"] = subset_df["id"].map(lambda x: ", ".join(id_to_found_keywords[x]))
		subset_df.sort_values(by=["num_found","id"], ascending=[False,True], inplace=True)
		subset_df.index = np.arange(1, len(subset_df)+1)

	if subset_df.shape[0] == 0:
		st.markdown("No genes were found for '{}'. Make sure the keywords and keyphrases in this search are separated by commas.".format(search_kws))
	else:
		# Describe what the results of the search are and what they mean in markdown.
		keywords_str = ", ".join([kw for kw in keywords if len(kw.split())==1])
		phrases_str = ", ".join([kw for kw in keywords if len(kw.split())>1])
		st.markdown("**Keyword(s)**: {}".format(keywords_str))
		st.markdown("**Phrase(s)**: {}".format(phrases_str))
		st.markdown("### Genes with Matching Phenotype Descriptions")
		st.markdown("The dataset of plant genes below shows only genes with phenotype description that included one or more of the searched keywords or phrases.")

		# Display the sorted and filtered dataset as a table with the relevant columns.
		st.table(data=subset_df[["Keywords", "Species", "Gene", "Gene Model", "Phenotype Description"]])

	# No need to keep this subset of the dataframe in memory if another search is performed.
	subset_df = None











################ Searching by PHENOTYPE DESCRIPTION ################ 

elif search_type == "phenotype" and input_text != "":

	# Start the results section of the page.
	st.markdown("## Results")

	# Do the processing of the search and add necessary columns to the dataframe to be shown.
	with st.spinner("Searching dataset for similar phenotype descriptions..."):
		search_string = input_text
		f_tokenizing = APPROACH_NAMES_AND_DATA[approach]["tokenization_function"]
		f_preprocessing = APPROACH_NAMES_AND_DATA[approach]["preprocessing_fucntion"]
		gene_id_to_result_string, gene_id_to_min_distance =  description_search(search_string, graph, f_tokenizing, f_preprocessing, len(RESULT_COLUMN_STRING), MAX_LINES_IN_RESULT_COLUMN)
		

		gene_id_to_result_string = {k:str(v) for k,v in gene_id_to_result_string.items()}



		df[RESULT_COLUMN_STRING] = df["id"].map(gene_id_to_result_string)

		#df[RESULT_COLUMN_STRING] = df[RESULT_COLUMN_STRING].map(lambda x: x.replace(r"\n","<br>"))




		df["distance"] = df["id"].map(gene_id_to_min_distance)
		df.sort_values(by=["distance","id"], ascending=[True,True], inplace=True)
		df.index = np.arange(1, len(df)+1)


	# Describe what the results of the search are and what they mean in markdown.
	st.markdown("### Genes with Matching Phenotype Descriptions")
	st.markdown("Genes with phenotypes that are described most similarity to '{}'".format(search_string))

	# Display the sorted and filtered dataset as a table with the relevant columns.
	#st.table(data=df[[RESULT_COLUMN_STRING, "Species", "Gene", "Gene Model", "Phenotype Description"]])
	#st.dataframe(df[[RESULT_COLUMN_STRING, "Species", "Gene", "Gene Model", "Phenotype Description"]])




	# NO MODULE NAME PLOTLY, ADD THIS THE CONDA ENVIRONMENT IF USING.
	import plotly
	import plotly.graph_objects as go







	def set_styles(results):
		table_styles = [
			dict(
				selector="table",
				props=[("font-size", "150%"), ("text-align", "center"), ("color", "red")],
			),
			dict(selector="caption", props=[("caption-side", "bottom")]),
		]
		#results.style.set_table_styles(table_styles).set_properties(**{"background-color": "blue", "color": "white"}).set_caption("This is a caption")
		results.style.set_table_styles(table_styles)
		return(results)






	my_df = df[[RESULT_COLUMN_STRING, "Species", "Gene", "Gene Model", "Phenotype Description"]]



	my_df = set_styles(my_df)


	#my_df = set_styles(my_df)



	header_values = my_df.columns

	#header_values = ["something with a <br> line <mark> break </mark>", "b", "c", "d", "e"]


	cell_values = []
	for index in range(0, len(my_df.columns)):
		cell_values.append(my_df.iloc[:,index:index+1])




	# Shouldn't have to do it this way, but we do. There is a bug with inserting the <br> tags any other way than in strings specified in this way.
	# For some reason, HTML tags present before this point are not recognized, I haven't figured out why.
	cell_values[0][RESULT_COLUMN_STRING] = cell_values[0][RESULT_COLUMN_STRING].map(lambda x: x.replace(r"\n\n",r"<br>"))
	descriptions = list(cell_values[0][RESULT_COLUMN_STRING].values)
	descriptions = [x.replace(r"\n\n","<br>") for x in descriptions]
	descriptions = [x.replace("all","<b>all</b>") for x in descriptions]
	cell_values[0] = descriptions


	#cell_values[0][RESULT_COLUMN_STRING] = cell_values[0][RESULT_COLUMN_STRING].astype(str)

	#cell_values[0] = ["aoiuser <br> waasdfs","dsdef"]










	style = True

	if not style:
		fig = go.Figure(
		   style_cell={
				'whiteSpace': 'normal',
				'height': 'auto',
			},
			data=[
				go.Table(
					header=dict(values=header_values), cells=dict(values=cell_values)
				)
			]
		)

	else:
		fig = go.Figure(
			#style_cell={
			#    'whiteSpace':'normal',
			#    'height': 'auto',
			#},
			data=[
				go.Table(

					columnorder = [1,2,3,4,5],
					columnwidth = [2,1,1,1,7],
					header=dict(values=header_values, fill_color="paleturquoise", align="left", font=dict(color='white', size=16), height=60),
					#cells=dict(values=cell_values, fill_color="lavender", align="left", font_size=14, height=40, line=dict(width=3)),
					cells=dict(values=cell_values, fill_color="lavender", align="left", font_size=14),
				)
			]
		)




	fig.update_layout(width=table_width, height=4000)

	st.plotly_chart(fig)
	






# Nothing was searched. Default to now showing anything and waiting for a widget value to change.
else:
	pass
	# TODO Should the whole dataset be displayed here instead?
	# TODO Keeping the else pass for now to remind me something might need to go here.













