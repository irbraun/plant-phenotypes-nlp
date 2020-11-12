import streamlit as st
import pandas as pd
import numpy as np
import sys
import re
import itertools
import nltk
import re
import base64
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import defaultdict
from string import punctuation
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import strip_non_alphanum, stem_text, preprocess_string, strip_tags, strip_punctuation
from PIL import Image
from textwrap import wrap
import plotly
import plotly.graph_objects as go


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








# Constants that help define how the tables look and how the text wraps within the table cells.
TABLE_HEADER_COLOR = "#808080"
TABLE_ROWS_COLOR = "#F1F2F6"
TABLE_HEIGHT = 1500
HEADER_HEIGHT = 30
RESULT_COLUMN_WIDTH = 55
MAX_LINES_IN_RESULT_COLUMN = 100
DESCRIPTION_COLUMN_WIDTH = 90
NEWLINE_TOKEN = "[NEWLINE]"
DIRECT_ANNOTATION_TAG = "Direct Annotation"
INHERITED_ANNOTATION_TAG = "Inherited Annotation"









# Constants that help define how the columns appear in the plottly tables. 
# The first value is the universal string key used throughout the script, so leave that alone.
# The second is how the column is titled in the presented tables so that can be changed just here and the change will take effect throughout.
# The third is the relative width of the column to all the other columns. Leave the rank column as 1 (the smallest), and change all othere
# with respect to that column.
column_info = [
	("rank", "Rank", 1),
	("score", "Score", 1),
	("result", "Result", 1),
	("keywords", "Query Keywords", 8),
	("sentences", "Query Sentences", 8),
	("terms", "Ontology Terms", 6),
	("species", "Species", 2),
	("gene", "Gene", 3),
	("model", "Gene Model", 3),
	("phenotype", "Phenotype Description", 12)
]
COLUMN_NAMES = {x[0]:x[1] for x in column_info}
COLUMN_WIDTHS = {x[0]:x[2] for x in column_info}









# Dangerous part. These have to exactly match how the text is treated within the notebook that generates the pairwise distances.
# There's no explicit check in the code that makes sure the processing is identical between those two locations.
# In the future this should probably be changed so both sets of code are calling one resources that knows how to do these things.
sentence_tokenize = lambda text: sent_tokenize(text)
as_one_token = lambda text: [text]
identify_function = lambda text: text
simple_preprocessing = lambda text: " ".join(simple_preprocess(text))
full_preprocessing = lambda text: " ".join(preprocess_string(text))




# Where are the dataset and ontology files that should be loaded for this application?
# The order of how the ontology names are given should exactly match the order of the ontology files.
DATASET_PATH = "resources/genes_texts_annots.csv"
ONTOLOGY_NAMES = ["PATO","PO","GO"]
ONTOLOGY_OBO_PATHS = ["resources/pato.obo", "resources/po.obo", "resources/go.obo"]
ONTOLOGY_PICKLE_PATHS = ["resources/pato.pickle", "resources/po.pickle", "resources/go.pickle"]





# These mappings are necessary if the internal strings used for species and how they should be displayed are different.
internal_species_strings = ["ath", "zma", "sly", "gmx", "osa", "mtr"]
display_species_strings = ["Arabidopsis", "Maize", "Tomato", "Soybean", "Rice", "Medicago"]
TO_SPECIES_DISPLAY_NAME = {i:d for i,d in zip(internal_species_strings,display_species_strings)}






# What are the different fields for each approach in this nested dictionary?
# path: The path to the pickle for loading the oats object associated with this approach.
# mapping_file: The paths to the pickle that is a dictionary that is needed to convert from the IDs for the saved vectors back to the gene IDs in this dataset.
# tokenization_function: A function for how text should be tokenized in order to be compatible with this approach.
# preprocessing_function: A function for how text should be preprocessed in order to be compatible with this approach.
APPROACH_NAMES_AND_DATA = {
	"n-grams":{
		"path_dists":"resources/dists_with_n_grams_tokenization_full_words_1_grams_tfidf.pickle", 
		"path_vectors":"resources/vectors_with_n_grams_tokenization_full_words_1_grams_tfidf.pickle",
		"mapping_file":"resources/gene_id_to_unique_ids_sent_tokens.pickle",
		"tokenization_function":sentence_tokenize,
		"preprocessing_function":full_preprocessing,
		},
	"plants":{
		"path_dists":"resources/dists_with_word2vec_tokenization_plants_size_300_mean.pickle", 
		"path_vectors":"resources/vectors_with_word2vec_tokenization_plants_size_300_mean.pickle",
		"mapping_file":"resources/gene_id_to_unique_ids_sent_tokens.pickle",
		"tokenization_function":sentence_tokenize,
		"preprocessing_function":full_preprocessing,
		},
	"wikipedia":{
		"path_dists":"resources/dists_with_word2vec_tokenization_wikipedia_size_300_mean.pickle", 
		"path_vectors":"resources/vectors_with_word2vec_tokenization_wikipedia_size_300_mean.pickle",
		"mapping_file":"resources/gene_id_to_unique_ids_sent_tokens.pickle",
		"tokenization_function":sentence_tokenize,
		"preprocessing_fucntion":identify_function,
		},
	}

from gensim.models.callbacks import CallbackAny2Vec
class LossLogger(CallbackAny2Vec):
    def __init__(self):
        self.epochs = []
        self.epoch = 1
        self.losses = []
        self.deltas = []
    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 1:
            delta = loss
        else:
            delta = loss- self.loss_previous_step
        self.loss_previous_step=loss
        self.losses.append(loss)
        self.epochs.append(self.epoch)
        self.epoch += 1
        self.deltas.append(delta)



# For testing, be able to subset this nested dictionary without having to uncomment sections of it.
# Just uncomment these two lines to use the entire set of approaches and load all files.
names_to_actually_use = ["plants","n-grams"]
APPROACH_NAMES_AND_DATA = {k:v for k,v in APPROACH_NAMES_AND_DATA.items() if k in names_to_actually_use}
SCORE_SIMILARITY_THRESHOLDS = {"plants":0.50, "n-grams":0.00}



# How should keywords and phrases be cleaned and handled as far as preprocessing or stemming goes?
KEYWORD_DELIM = "[DELIM]"
KEYWORD_PREPROCESSING_FILTERS = [lambda x: x.lower(), strip_non_alphanum, strip_tags, strip_punctuation, stem_text]
PREPROCESSING_FOR_KEYWORD_SEARCH_FUNCTION = lambda x: "{}{}{}".format(KEYWORD_DELIM, KEYWORD_DELIM.join([token for token in preprocess_string(x, KEYWORD_PREPROCESSING_FILTERS)]), KEYWORD_DELIM)



















# Initial configuration and the logo image at the top of the page.
st.set_page_config(page_title="QuOATS", layout="wide", initial_sidebar_state="expanded")
PATH_TO_LOGO_PNG = "resources/logo_without_box.png"
st.image(Image.open(PATH_TO_LOGO_PNG), caption=None, width=800, output_format="png")

# Markdown for introducing the app and linking to other relevant resources like the project Github page.


#st.markdown("# QuOATS")

st.markdown("## Documentation")

show_documentation = st.checkbox(label="Show", value=False)




documentation_string = """

This tool enables querying a dataset of plant genes using annotations with ontology terms and natural language. The dataset and the
motivation behind this tool are described in detail in the preprint that is available here [add link to the preprint here]. The tool
provides four different methods querying the dataset, and the result in each case is a table of genes sorted by relevance to the query.
The actions behind each methods and the nature of the returned results are described in detail in the next sections. The results of any 
query can be downloaded as a tsv file using the provided link.


### Gene Identifiers
The genes in this dataset are associated with several types of identifying strings. These include gene names, gene
symbols, protein names, gene models (e.g., AT3G17980, GRMZM2G422750), and other aliases and synonyms. To use this 
search method, enter a string into the searchbar, and if there are any matches in the dataset, a list of one or more 
matching genes will be brought up. Check the 'show possible gene synonyms' option to the left to show other gene identifiers
associated with each gene returned by a search. Selecting one of the returned genes will execute a free text query using
the phenotype description associated with that gene to identify other genes that are associated with similarly described
phenotypes. For details on that query, see the 'Free Text' section below. The number of genes that are in the returned
table can be adjusted to the left. Uncheck the 'compress phenotypes in table' option to display the entirety of the 
phenotype descriptions associated with each gene.

### Ontology Terms
Genes in this dataset have been annotated by curators with ontology terms from the Gene Ontology (GO), the Plant Ontology (PO), 
and the Phenotype and Trait Ontology (PATO). This search option can be used to identify which genes in the dataset are associated
with which ontology terms. Enter any number of ontology term IDs, separated by either commas or spaces. The term IDs should be in
standard format with the ontology name and the term number (e.g., PATO:0000587). The table of returned genes contains the genes
that were either directly annotated with the searched terms, or inherited this annotation through the ontology hierarchy, which is
indicated in the results. The genes are sorted based on how many of the searched terms were direct annotations, and then how many 
of the searched terms were inherited annotations. Uncheck the 'compress phenotypes in table' option to display the entirety of the 
phenotype descriptions associated with each gene.

### Keywords & Keyphrases
This search option identifies genes in the dataset that have phenotype descriptions that contain particular words or phrases.
Enter any number of search strings separated by commas. The results of this search indicate which words or phrases are present
in the phenotype description of each returned gene. The genes are ordered based on the quantity of searched words or phrases 
present in their phenotype descriptions. Although this search option is for exact matching, preprocessing and normalization 
of the search text is done to ensure that words with the same stem or that differ only by case still match (e.g., plants, plant, and Plant). 
Uncheck the 'compress phenotypes in table' option to display the entirety of the phenotype descriptions associated with each gene.

### Free Text
This search option allows for querying with a phenotype descriptions or set of phenotype descriptions (separated by periods), in
order to recover genes that are associated with phenotypes that have been similarly described. Rather than using exact matching
only, this search type combines text embedding models that are capable of making associations between related words or concepts.
Each string separated by periods in the query is compared to each sentence or fragment in the phenotype descriptions of genes in
the dataset using two different NLP methods (n-grams, and word embedding models trained on phenotype descriptions and plant-related
abstracts). The similarities are average across methods, and for each gene, the greatest similarity between text in that 
gene's phenotype description and the queried text is returned. The returned genes are ranked based on the greatest individual 
similarity and then the average similarity across all descriptions provided in the query. Place square brackets around any word
or set of words in the queried sentences in order to only consider matches where the bracketed text is present, similar to the 
keyword search described above. Uncheck the 'compress phenotypes in table' option to display the entirety of the phenotype descriptions associated with each gene.


"""









if show_documentation:
	st.markdown(documentation_string)









st.markdown("## Quick Overview")


overview_string = """
 - Search using **gene identifiers** by entering in a gene name, protein name, gene model, or other identifer. For example, 'ATG7'.
 - Search using **ontology terms** by entering ontology term IDs separeted by commas or spaces. For example, 'PATO:0000587, GO:0010029, PO:0020127'.
 - Search using **keywords or keyphrases** by entering any strings separated by commas, e.g., "height, root system, auxin". Only phenotypes containing those concepts are returned.
 - Search using **free text** by entering any words or phenotype descriptions separated by periods, e.g., "leaves are wider than normal. "resistant to 
 infection." Returned phenotypes are scored with respect to each query component and may be related in meaning but not identically described. Place square brackets around a particular word to only return phenotypes that explicity contain it, e.g., "resistant to [bacterial] infection."


"""


st.markdown(overview_string)
























#B31334
#E7FD8E
#87F1CC







# Setting some of the color schemes and formatting of the page.
st.markdown(
	"""
	<style>
	body {
		color: #111;
		background-color: #fff;
	}
	.reportview-container .markdown-text-container {
		font-family: arial;
	}
	.sidebar .sidebar-content {
		background-image: linear-gradient(#E7FD8E,#E7FD8E);
		color: black;
	}
	.Widget>label {
		color: black;
		font-family: arial;
	}
	[class^="st-b"]  {
		color: black;
		font-family: arial;
	}
	.st-bb {
		background-color: transparent;
	}
	.st-at {
		background-color: transparent;
	}
	footer {
		color: black;
		font-family: times;
	}
	.reportview-container .main footer, .reportview-container .main footer a {
		color: #fff;
	}
	header .decoration {
		background-image: none;
	}
	""",
	unsafe_allow_html=True,
)














# Note that the output object(s) and dictionaries should not actually be modified in this script.
# But streamlit was detecting a change in the outputs of this function? So the allow output mutation
# decorator is necessary in order for this cached function to not be run again on refresh.
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def read_in_files(dataset_path, approach_names_and_data):
	"""Read in the large files that initialize the application here, which takes a long time.
	
	Args:
	    dataset_path (TYPE): Description
	    approach_names_and_data (TYPE): Description
	
	Returns:
	    TYPE: Description
	"""
	
	# Read in the dataset and create its object.
	dataset = Dataset(data=dataset_path, keep_ids=True)
	dataset.filter_has_description()

	# Reading in the large files that store objects for vectorizing and comparing texts.
	approach_to_object = {}
	approach_to_mapping = {}
	for k,v in approach_names_and_data.items():
		obj = load_from_pickle(v["path_dists"])
		vectors = load_from_pickle(v["path_vectors"])
		obj.vector_dictionary = vectors
		mapping = load_from_pickle(v["mapping_file"])
		approach_to_mapping[k] = mapping
		approach_to_object[k] = obj

	# Other expensive computation here so that it's done inside the cached function.
	# Get the dataframe for this dataset and add some additional columns that will be useful for displaying information.
	# Slow additions to the dataframe should go here. Very fast ones can go in the main non-cached part of the script.
	df = dataset.to_pandas()
	df[COLUMN_NAMES["gene"]] = df["id"].map(lambda x: dataset.get_gene_dictionary()[x].primary_identifier)
	df[COLUMN_NAMES["model"]] = df["id"].map(lambda x: dataset.get_gene_dictionary()[x].gene_models[0] if len(dataset.get_gene_dictionary()[x].gene_models)>0 else "")
	df["descriptions_one_line_truncated"] = df["descriptions"].map(lambda x: truncate_string(x, DESCRIPTION_COLUMN_WIDTH))
	df["descriptions_with_newline_tokens"] = df["descriptions"].map(lambda x: NEWLINE_TOKEN.join(wrap(x, DESCRIPTION_COLUMN_WIDTH)))

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
	"""Return a truncated version of the text and adding elipses if it's longer than the character limit.
	
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







def format_result_strings(query_sentences, gene_ids, gene_id_to_distances, result_column_width, result_column_max_lines):
	"""This takes the lists of query sentences score against the existing data and formats wrapped strings for the table.
	
	Args:
	    query_sentences (TYPE): Description
	    gene_ids (TYPE): Description
	    gene_id_to_distances (TYPE): Description
	    result_column_width (TYPE): Description
	    result_column_max_lines (TYPE): Description
	
	Returns:
	    TYPE: Description
	"""
	# The list of query sentences should correspond to the list of distances that each gene ID is mapped to.
	# Have to trust that these are in corresponding order, that happens in another section.
	assert len(query_sentences) == len(list(gene_id_to_distances.values())[0])

	# Subsetting the gene ID to distances dictionary in case we don't need to format everything, speed things up.
	gene_id_to_distances = {gene_id:gene_id_to_distances[gene_id] for gene_id in gene_ids}


	# Making the formatted strings for the sentence tokens from the query and the similarity scores of each of them.
	gene_id_to_formatted_queries  = {}
	gene_id_to_formatted_similarities = {}
	truncated_query_sents = [truncate_string(s, result_column_width) for s in query_sentences]
	for gene_id, distances in gene_id_to_distances.items():

		sorted_queries_and_distances = sorted(zip(truncated_query_sents, distances), key=lambda x:x[1])[0:result_column_max_lines]
		sorted_queries = [x[0] for x in sorted_queries_and_distances]
		sorted_distances = [x[1] for x in sorted_queries_and_distances]
		sorted_similarities = ["{:.2f}".format(1-x) for x in sorted_distances]
		formatted_query_string = NEWLINE_TOKEN.join(sorted_queries)
		formatted_similarities_string = NEWLINE_TOKEN.join(sorted_similarities)
		gene_id_to_formatted_queries[gene_id] = formatted_query_string
		gene_id_to_formatted_similarities[gene_id] = formatted_similarities_string

	return(gene_id_to_formatted_queries, gene_id_to_formatted_similarities)







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
	species_to_gene_id_list = {s:list(set(l)) for s,l in species_to_gene_id_list.items()}
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





def ontology_term_search(id_to_direct_annotations, id_to_indirect_annotations, term_ids):
	"""Helper function for searching the dataset for ontology term annotations.
	
	Args:
	    id_to_direct_annotations (TYPE): Description
	    id_to_indirect_annotations (TYPE): Description
	    term_ids (TYPE): Description
	
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
		num_direct = 0
		num_indirect = 0
		lines = []
		for idx,term_id in enumerate(term_ids):
			if gene_id_to_direct_match[gene_id][idx]:
				match_type = DIRECT_ANNOTATION_TAG
				num_direct = num_direct+1
				match_str = "({})".format(match_type)
				term_id_str = term_id
				line = "{} {}".format(term_id_str, match_str)
				lines.append(line)
			elif gene_id_to_indirect_match[gene_id][idx]:
				match_type = INHERITED_ANNOTATION_TAG
				num_indirect = num_indirect+1
				match_str = "({})".format(match_type)
				term_id_str = term_id
				line = "{} {}".format(term_id_str, match_str)
				lines.append(line)
			else:
				continue

		gene_id_to_num_direct_matches[gene_id] = num_direct
		gene_id_to_num_indirect_matches[gene_id] = num_indirect
		result_str = NEWLINE_TOKEN.join(lines)
		gene_id_to_result_string[gene_id] = result_str
		assert len(lines) == gene_id_to_num_direct_matches[gene_id]+gene_id_to_num_indirect_matches[gene_id]

	return(gene_id_to_num_direct_matches, gene_id_to_num_indirect_matches, gene_id_to_result_string)






def description_search(text, tokenization_function):
	"""Helper function for searching the dataset for similar phenotype descriptions.
	
	Args:
	    text (TYPE): Description
	    graph (TYPE): Description
	    tokenization_function (TYPE): Description
	    preprocessing_function (TYPE): Description
	
	Returns:
	    TYPE: Description

	
	"""
	# Do tokenization and preprocessing on the searched text to yield a list of strings.
	# The tokenization and preprocessing have to match the object that will be used, and this decision is informed by
	# some mapping that is already done, and the choices are passed in to this function.
	# The tokenization might just yield a single text, but it still goes in a list for consistency.
	sentence_tokens = tokenization_function(text)





	# Adding something here.
	list_of_keywords_for_each_sentence = []
	pattern = re.compile(r"\[(.*?)\]")
	for s in sentence_tokens:
		keywords = pattern.findall(s)
		list_of_keywords_for_each_sentence.append(keywords)
	sentence_tokens = [s.replace("[","").replace("]","") for s in sentence_tokens]





	gene_id_to_approach_to_distances = defaultdict(lambda: defaultdict(list))
	for approach in APPROACHES:


		preprocessing_function = APPROACH_NAMES_AND_DATA[approach]["preprocessing_function"]
		preprocessed_sentence_tokens  = [preprocessing_function(s) for s in sentence_tokens]
		graph = APPROACH_TO_OBJECT[approach]
		gene_id_to_graph_ids = APPROACH_TO_MAPPING[approach]


		# Now the preprocessed sentence tokens are in the right format, and ready to be embedded and compared to the existing data.
		# Get a mapping between gene IDs and their distances to each new text string parsed from the search string.
		# The key is the gene ID from the existing dataset and the values are a list in the same order as the preprocessed query sentences.
		#gene_id_to_distances = defaultdict(list)
		for s,keyword_list in zip(preprocessed_sentence_tokens, list_of_keywords_for_each_sentence):
			
			internal_id_to_distance_from_new_text = graph.get_distances(s)


			# Check for the keywords, and force those distances for sentences in the dataset where a part of that keyword doesn't appear.
			if len(keyword_list)>0:
				keyword_list = [preprocessing_function(s) for s in keyword_list]

				for keyword in keyword_list:
					internal_id_to_distance_from_keyword = APPROACH_TO_OBJECT["n-grams"].get_distances(keyword)
					internal_ids_where_the_keyword_score_is_zero = [i for i,dist in internal_id_to_distance_from_keyword.items() if dist==1.00]
					for i in internal_ids_where_the_keyword_score_is_zero:
						internal_id_to_distance_from_new_text[i] = 1.00




			for gene_id,graph_ids in gene_id_to_graph_ids.items():
				# What's the smallest distances between this new graph and one of the texts for the internal graph nodes.
				graph_distances = [internal_id_to_distance_from_new_text[graph_id] for graph_id in graph_ids]
				min_graph_distance = min(graph_distances)

				# Tranforming distances scores if necessary to discount similarities below a certain threshold depending on the method.
				# Makes results displayed more intuitive and limites the output tables from being unreasonably large.
				x = min_graph_distance
				x = 1-x
				spread_after_thresholding = 1-SCORE_SIMILARITY_THRESHOLDS[approach]
				x = (max(x,SCORE_SIMILARITY_THRESHOLDS[approach])-SCORE_SIMILARITY_THRESHOLDS[approach])/spread_after_thresholding
				x = 1-x
				min_graph_distance = x


				# Remember that value as the distance from this one parsed text string from the search to this particular gene.
				#gene_id_to_distances[gene_id].append(min_graph_distance)
				gene_id_to_approach_to_distances[gene_id][approach].append(min_graph_distance)




	# Where the distances from multiple methods get combined. Should be either min() or mean() here. These are used for sorting.
	gene_id_to_combined_distances = defaultdict(list)
	for gene_id in gene_id_to_approach_to_distances.keys():
		distance_lists = []
		for approach in APPROACHES:
			distance_lists.append(gene_id_to_approach_to_distances[gene_id][approach])
		distance_lists = np.array(distance_lists)
		combined_distances = np.min(distance_lists, axis=0)
		gene_id_to_combined_distances[gene_id] = list(combined_distances)

	gene_id_to_min_distance = {gene_id:min(distances) for gene_id,distances in gene_id_to_combined_distances.items()}
	gene_id_to_mean_distance = {gene_id:np.mean(distances) for gene_id,distances in gene_id_to_combined_distances.items()}


	return(sentence_tokens, gene_id_to_combined_distances, gene_id_to_min_distance, gene_id_to_mean_distance)
















# The first dictionary created here maps the name of an approach, like "n-grams" to an oats.distances object.
# The second dictionary created here maps the name of an approach like "n-grams" to another dictionary that
# maps the IDs that in this dataset to the IDs that are used internally by that corresponding distances object.
# This is necessary because those internal IDs represent compressed or altered versions of the text and could 
# refer to more than one text instance in the actual dataset that is seen here.
with st.spinner("Reading very large files, this might take a few minutes."):
	dataset, approach_to_object, approach_to_mapping, df, id_to_descriptions_for_keyword_matching = read_in_files(DATASET_PATH, APPROACH_NAMES_AND_DATA)
	ontologies = read_in_ontologies(ONTOLOGY_NAMES, ONTOLOGY_PICKLE_PATHS)



APPROACHES = list(approach_to_object.keys())
APPROACH_TO_MAPPING = approach_to_mapping
APPROACH_TO_OBJECT = approach_to_object














############# The Sidebar ###############

# Presenting the options for filtering by species.
st.sidebar.markdown("### Filtering by Species")
species_display_names = [TO_SPECIES_DISPLAY_NAME[x] for x in dataset.get_species()]
species_list = st.sidebar.multiselect(label="Filter to only include certain species", options=species_display_names)
if len(species_list) == 0:
	species_list = species_display_names

# Present information and search options for finding a particular gene or protein in the sidebar.
#st.sidebar.markdown("### Select a Similarity Algorithm")
#approach = st.sidebar.selectbox(label="Select one", options=list(approach_to_object.keys()), index=0)


# Presenting the general options for how the data is displayed.
st.sidebar.markdown("### General Page Options")
truncate = st.sidebar.checkbox(label="Compress phenotypes in table", value=True)
synonyms = st.sidebar.checkbox(label="Show possible gene synonyms", value=False)
include_examples = st.sidebar.checkbox(label="Show example queries", value=False)
ROW_LIMIT = st.sidebar.number_input("Number of genes to display in results", min_value=1, max_value=None, value=50, step=50)
TABLE_WIDTH = st.sidebar.slider(label="Table Width (Pixels)", min_value=400, max_value=8000, value=2000, step=100, format=None, key=None)













ref_text = """
	If you find this tool useful in your own research, please cite [add reference to preprint here].
	"""

contact_text = """
	If you have feedback about the tool or experience issues while using it, please contact irbraun@iastate.edu or szarecor@iastate.edu.
	"""

st.sidebar.markdown("### Citation")
st.sidebar.markdown(ref_text)
st.sidebar.markdown("### Contact")
st.sidebar.markdown(contact_text)



















############# The Searchbar Section ###############


# Display the search section of the main page.
st.markdown("## Search")

search_types = ["gene", "ontology", "keyword", "phenotype"]
search_types_labels = ["Gene Identifiers", "Ontology Terms", "Keywords & Keyphrases", "Free Text"]
search_type_examples = ["epc1", "PATO:0000587, GO:0010029, PO:0020127", "dwarfism, root system, leaves, auxin", "Plants are reduced in height. Plants have wide leaves. Smaller than normal. Something else too."]

if include_examples:
	search_type_label_map = {t:"{} (e.g. '{}')".format(l,e) for t,l,e in zip(search_types, search_types_labels, search_type_examples)}
else:
	search_type_label_map = {t:l for t,l in zip(search_types,search_types_labels)}


search_types_format_func = lambda x: search_type_label_map[x]
search_type = st.radio(label="Select a type of search", options=search_types, index=0, format_func=search_types_format_func)
input_text = st.text_input(label="Enter text here")
















############## Default view, presenting the entire dataset ################



# Subset the dataframe if necessary.

# The dataframe as it should be formatted for displaying in the app, not all data is needed.
df["Species"] = df["species"].map(TO_SPECIES_DISPLAY_NAME)
df = df[df["Species"].isin(species_list)]





def display_download_link(df, column_keys, column_keys_to_unwrap, column_keys_to_list, num_rows):

	# Subsetting the dataframe.
	my_df = df[[COLUMN_NAMES[x] for x in column_keys]].head(num_rows)


	# Anything that had newline characters that were being used just to wrap to the next line.
	for key in column_keys_to_unwrap:
		my_df[COLUMN_NAMES[key]] = my_df[COLUMN_NAMES[key]].map(lambda x: x.replace(NEWLINE_TOKEN,""))

	# Anything where the newline character was being used to separate a list of items like scores of terms.
	for key in column_keys_to_list:
		my_df[COLUMN_NAMES[key]] = my_df[COLUMN_NAMES[key]].map(lambda x: "{}".format("; ".join(x.split(NEWLINE_TOKEN))))


	# Presenting a download link for a csv file.
	#csv = my_df.to_csv(index=False)
	#b64 = base64.b64encode(csv.encode()).decode() 
	#link = f'<a href="data:file/csv;base64,{b64}" download="query_results.csv">Download (CSV)</a>'
	#st.markdown(link, unsafe_allow_html=True)

	# Presenting a download link for a tsv file.
	tsv = my_df.to_csv(index=False, sep="\t")
	b64 = base64.b64encode(tsv.encode()).decode() 
	link = f'<a href="data:file/tsv;base64,{b64}" download="query_results.tsv">Download tsv file</a>'
	st.markdown(link, unsafe_allow_html=True)







def display_plottly_dataframe(df, column_keys, column_keys_to_wrap, num_rows):

	my_df = df[[COLUMN_NAMES[x] for x in column_keys]].head(num_rows)

	header_values = my_df.columns
	cell_values = []
	for index in range(0, len(my_df.columns)):
		cell_values.append(my_df.iloc[:,index:index+1])


	# Shouldn't have to do it this way, but we do. There is a bug with inserting the <br> tags any other way than in strings specified in this way.
	# For some reason, HTML tags present before this point are not recognized, I haven't figured out why.
	indices_of_columns_to_wrap = [column_keys.index(x) for x in column_keys_to_wrap]
	for col_idx,col_key in zip(indices_of_columns_to_wrap,column_keys_to_wrap):
		contents = list(cell_values[col_idx][COLUMN_NAMES[col_key]].values)
		contents = [x.replace(NEWLINE_TOKEN,"<br>") for x in contents]
		cell_values[col_idx] = contents


	fig = go.Figure(data=[go.Table(
		columnorder = list(range(len(column_keys))),
		columnwidth = [COLUMN_WIDTHS[x] for x in column_keys],
		header=dict(values=header_values, fill_color=TABLE_HEADER_COLOR, align="left", font=dict(color='black', size=14), height=HEADER_HEIGHT),
		cells=dict(values=cell_values, fill_color=TABLE_ROWS_COLOR, align="left", font=dict(color='black', size=14)),
		)])

	fig.update_layout(width=TABLE_WIDTH, height=TABLE_HEIGHT)
	st.plotly_chart(fig)









def truncate_description_cell(s, num_lines_to_keep):
	lines = s.split(NEWLINE_TOKEN)
	if len(lines) > num_lines_to_keep:
		index_of_last_line = num_lines_to_keep-1
		lines[num_lines_to_keep-1] = "{}...".format(lines[num_lines_to_keep-1])
		modified_s = NEWLINE_TOKEN.join(lines[:index_of_last_line+1])
		return(modified_s)
	else:
		return(s)









############### Listening for something to typed and entered into any of the search boxes #############











#    ********  ******** ****     ** ********   ** *******   ******** ****     ** ********** ** ******** ** ******** *******  
#   **//////**/**///// /**/**   /**/**/////   /**/**////** /**///// /**/**   /**/////**/// /**/**///// /**/**///// /**////** 
#  **      // /**      /**//**  /**/**        /**/**    /**/**      /**//**  /**    /**    /**/**      /**/**      /**   /** 
# /**         /******* /** //** /**/*******   /**/**    /**/******* /** //** /**    /**    /**/******* /**/******* /*******  
# /**    *****/**////  /**  //**/**/**////    /**/**    /**/**////  /**  //**/**    /**    /**/**////  /**/**////  /**///**  
# //**  ////**/**      /**   //****/**        /**/**    ** /**      /**   //****    /**    /**/**      /**/**      /**  //** 
#  //******** /********/**    //***/********  /**/*******  /********/**    //***    /**    /**/**      /**/********/**   //**
#   ////////  //////// //      /// ////////   // ///////   //////// //      ///     //     // //       // //////// //     // 



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
		unique_button_key = 1
		for species,id_list in gene_matches.items():
			for i in id_list:
				primary_gene_name = dataset.get_gene_dictionary()[i].primary_identifier
				other_names = dataset.get_gene_dictionary()[i].all_identifiers
				button_label = "{}: {}".format(TO_SPECIES_DISPLAY_NAME[species], primary_gene_name)
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



				# The tokenization has to be the same for all the methods used, which in this case is just sentence tokenization. So just take one of them.
				# TODO make this more organized, don't specifyc tokenization function in the dictionary if it needs to be identical across used methods.
				f_tokenizing = APPROACH_NAMES_AND_DATA[APPROACHES[0]]["tokenization_function"]
				raw_sentence_tokens, gene_id_to_distances, gene_id_to_min_distance, gene_id_to_mean_distance = description_search(search_string, f_tokenizing)
				df["min_distance"] = df["id"].map(gene_id_to_min_distance)
				df["mean_distance"] = df["id"].map(gene_id_to_mean_distance)
				df.sort_values(by=["min_distance","mean_distance","id"], ascending=[True,True,True], inplace=True)
				df["Rank"] = np.arange(1, len(df)+1)
				df = df.head(ROW_LIMIT)
				gene_ids = df["id"].values
				gene_id_to_formatted_queries, gene_id_to_formatted_similarities = format_result_strings(raw_sentence_tokens, gene_ids, gene_id_to_distances, RESULT_COLUMN_WIDTH, MAX_LINES_IN_RESULT_COLUMN)


				df[COLUMN_NAMES["sentences"]] = df["id"].map(gene_id_to_formatted_queries)
				df[COLUMN_NAMES["score"]] = df["id"].map(gene_id_to_formatted_similarities)


				if truncate:
					df["Phenotype Description"] = df["descriptions_with_newline_tokens"]
					df["Phenotype Description"] = df["Phenotype Description"].map(lambda x: truncate_description_cell(x, min(len(raw_sentence_tokens),MAX_LINES_IN_RESULT_COLUMN)))

				else:
					df["Phenotype Description"] = df["descriptions_with_newline_tokens"]


			columns_to_include_keys = ["rank", "score", "sentences", "species", "gene", "model", "phenotype"]
			columns_to_include_keys_and_wrap = ["score", "sentences", "phenotype"]
			column_keys_to_unwrap = ["phenotype"]
			column_keys_to_list = ["score", "sentences"]
			display_download_link(df, columns_to_include_keys, column_keys_to_unwrap, column_keys_to_list, ROW_LIMIT)
			display_plottly_dataframe(df, columns_to_include_keys, columns_to_include_keys_and_wrap, ROW_LIMIT)




















#  ********** ******** *******   ****     ****  ********
# /////**/// /**///// /**////** /**/**   **/** **////// 
#     /**    /**      /**   /** /**//** ** /**/**       
#     /**    /******* /*******  /** //***  /**/*********
#     /**    /**////  /**///**  /**  //*   /**////////**
#     /**    /**      /**  //** /**   /    /**       /**
#     /**    /********/**   //**/**        /** ******** 
#     //     //////// //     // //         // ////////  


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


		gene_id_to_num_direct_matches, gene_id_to_num_indirect_matches, gene_id_to_result_string = ontology_term_search(direct_annotations, inherited_annotations, term_ids)
		df[COLUMN_NAMES["terms"]] = df["id"].map(gene_id_to_result_string)
		df["num_direct"] = df["id"].map(gene_id_to_num_direct_matches)
		df["num_indirect"] = df["id"].map(gene_id_to_num_indirect_matches)
		subset_df = df[(df["num_direct"]>0) | (df["num_indirect"]>0)]
		subset_df["num_either"] = subset_df["num_direct"]+subset_df["num_indirect"]
		subset_df.sort_values(by=["num_direct","num_indirect","id"], ascending=[False,False,True], inplace=True)
		subset_df[COLUMN_NAMES["result"]] = np.arange(1, len(subset_df)+1)


		if subset_df.shape[0] == 0:
			st.markdown("No genes were found for '{}'. Make sure the ontology term IDs are separated by commas or spaces and formatted like the examples.".format(term_id))
		else:
			# Describe what the results of the search are and what they mean in markdown.
			st.markdown("### Genes with Matching Annotations")
			st.markdown("The dataset of plant genes below shows only genes with annotations that included one or more of the searched ontology terms.")


			subset_df[COLUMN_NAMES["phenotype"]] = subset_df["descriptions_with_newline_tokens"]
			if truncate:
				subset_df[COLUMN_NAMES["phenotype"]] = subset_df[["descriptions_with_newline_tokens","num_either"]].apply(lambda row: truncate_description_cell(row["descriptions_with_newline_tokens"], min(row["num_either"],MAX_LINES_IN_RESULT_COLUMN)),axis=1)


			columns_to_include_keys = ["result", "terms", "species", "gene", "model", "phenotype"]
			columns_to_include_keys_and_wrap = ["terms", "phenotype"]
			column_keys_to_unwrap = ["phenotype"]
			column_keys_to_list = ["terms"]
			display_download_link(subset_df, columns_to_include_keys, column_keys_to_unwrap, column_keys_to_list, ROW_LIMIT)
			display_plottly_dataframe(subset_df, columns_to_include_keys, columns_to_include_keys_and_wrap, ROW_LIMIT)















#  **   ** ******** **    ** **       **   *******   *******   *******    ********
# /**  ** /**///// //**  ** /**      /**  **/////** /**////** /**////**  **////// 
# /** **  /**       //****  /**   *  /** **     //**/**   /** /**    /**/**       
# /****   /*******   //**   /**  *** /**/**      /**/*******  /**    /**/*********
# /**/**  /**////     /**   /** **/**/**/**      /**/**///**  /**    /**////////**
# /**//** /**         /**   /**** //****//**     ** /**  //** /**    **        /**
# /** //**/********   /**   /**/   ///** //*******  /**   //**/*******   ******** 
# //   // ////////    //    //       //   ///////   //     // ///////   ////////  

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
		subset_df[COLUMN_NAMES["keywords"]] = subset_df["id"].map(lambda x: ", ".join(id_to_found_keywords[x]))
		subset_df.sort_values(by=["num_found","id"], ascending=[False,True], inplace=True)
		subset_df[COLUMN_NAMES["result"]] = np.arange(1, len(subset_df)+1)

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

		# Formatting the columns correctly and truncating ones that wrap on multiple lines if the table is compressed.
		if truncate:
			subset_df[COLUMN_NAMES["phenotype"]] = subset_df["descriptions_one_line_truncated"]
		else:
			subset_df[COLUMN_NAMES["phenotype"]] = subset_df["descriptions_with_newline_tokens"]


		# Show the subset of columns that is relevant to this search.
		columns_to_include_keys = ["result", "keywords", "species", "gene", "model", "phenotype"]
		columns_to_include_keys_and_wrap = ["phenotype"]
		column_keys_to_unwrap = ["phenotype"]
		column_keys_to_list = []
		display_download_link(subset_df, columns_to_include_keys, column_keys_to_unwrap, column_keys_to_list, ROW_LIMIT)
		display_plottly_dataframe(subset_df, columns_to_include_keys, columns_to_include_keys_and_wrap, ROW_LIMIT)

	# No need to keep this subset of the dataframe in memory if another search is performed.
	subset_df = None











#  ******** *******   ******** ********   ********** ******** **     ** **********
# /**///// /**////** /**///// /**/////   /////**/// /**///// //**   ** /////**/// 
# /**      /**   /** /**      /**            /**    /**       //** **      /**    
# /******* /*******  /******* /*******       /**    /*******   //***       /**    
# /**////  /**///**  /**////  /**////        /**    /**////     **/**      /**    
# /**      /**  //** /**      /**            /**    /**        ** //**     /**    
# /**      /**   //**/********/********      /**    /******** **   //**    /**    
# //       //     // //////// ////////       //     //////// //     //     //     

elif search_type == "phenotype" and input_text != "":


	# Start the results section of the page.
	st.markdown("## Results")

	# Do the processing of the search and add necessary columns to the dataframe to be shown.
	with st.spinner("Searching dataset for similar phenotype descriptions..."):
		



		search_string = input_text








		# First check if want to carry out a keyword search first.
		pattern = re.compile(r"\[(.*?)\]")
		keywords = pattern.findall(search_string)
		if len(keywords)>0:
			raw_keywords = [kw.strip() for kw in keywords]
			modified_keywords = [PREPROCESSING_FOR_KEYWORD_SEARCH_FUNCTION(kw) for kw in raw_keywords]
			id_to_found_keywords, id_to_num_found_keywords = keyword_search(id_to_descriptions_for_keyword_matching, raw_keywords, modified_keywords)
			df["num_found"] = df["id"].map(id_to_num_found_keywords)
			subset_df = df[df["num_found"]>0]
			subset_df[COLUMN_NAMES["keywords"]] = subset_df["id"].map(lambda x: ", ".join(id_to_found_keywords[x]))
			subset_df.sort_values(by=["num_found","id"], ascending=[False,True], inplace=True)
			subset_df[COLUMN_NAMES["result"]] = np.arange(1, len(subset_df)+1)
			df = subset_df







		# The tokenization has to be the same for all the methods used, which in this case is just sentence tokenization. So just take one of them.
		# TODO make this more organized, don't specify tokenization function in the dictionary if it needs to be identical across used methods.
		f_tokenizing = APPROACH_NAMES_AND_DATA[APPROACHES[0]]["tokenization_function"]
		raw_sentence_tokens, gene_id_to_distances, gene_id_to_min_distance, gene_id_to_mean_distance = description_search(search_string, f_tokenizing)




		# Added this part to account for allowing brackets for keywords.
		#raw_sentence_tokens = [s.replace("[","").replace("]","") for s in raw_sentence_tokens]



		# Generate a column that will be used as the sorting metric, and sort the dataframe.
		df["min_distance"] = df["id"].map(gene_id_to_min_distance)
		df["mean_distance"] = df["id"].map(gene_id_to_mean_distance)
		df.sort_values(by=["min_distance","mean_distance","id"], ascending=[True,True,True], inplace=True)
		
		# Subset it from this point forward after sorting to not waste formatting time on something that won't be shown.
		min_distance_threshold = 0.99
		df = df[df["min_distance"]<min_distance_threshold]
		df = df.head(ROW_LIMIT)


		# Create all the formatted columns that will need to be displayed.
		df[COLUMN_NAMES["rank"]] = np.arange(1, len(df)+1)
		gene_ids = df["id"].values
		gene_id_to_formatted_queries, gene_id_to_formatted_similarities = format_result_strings(raw_sentence_tokens, gene_ids, gene_id_to_distances, RESULT_COLUMN_WIDTH, MAX_LINES_IN_RESULT_COLUMN)
		df[COLUMN_NAMES["sentences"]] = df["id"].map(gene_id_to_formatted_queries)
		df[COLUMN_NAMES["score"]] = df["id"].map(gene_id_to_formatted_similarities)

	# Describe what the results of the search are and what they mean in markdown.
	st.markdown("### Genes with Matching Phenotype Descriptions")
	st.markdown("Genes with phenotypes that are described most similarly to '{}'".format(search_string))


	# Formatting the columns correctly and truncating ones that wrap on multiple lines if the table is compressed.
	df[COLUMN_NAMES["phenotype"]] = df["descriptions_with_newline_tokens"]
	if truncate:
		df[COLUMN_NAMES["phenotype"]] = df["Phenotype Description"].map(lambda x: truncate_description_cell(x, min(len(raw_sentence_tokens),MAX_LINES_IN_RESULT_COLUMN)))






	# Show the subset of columns that is relevant to this search.
	columns_to_include_keys = ["rank", "score", "sentences", "species", "gene", "model", "phenotype"]
	columns_to_include_keys_and_wrap = ["score", "sentences", "phenotype"]
	column_keys_to_unwrap = ["phenotype"]
	column_keys_to_list = ["score","sentences"]
	display_download_link(df, columns_to_include_keys, column_keys_to_unwrap, column_keys_to_list, ROW_LIMIT)
	display_plottly_dataframe(df, columns_to_include_keys, columns_to_include_keys_and_wrap, ROW_LIMIT)







# Nothing was searched. Default to now showing anything and waiting for a widget value to change.
else:
	pass
	# TODO Should the whole dataset be displayed here instead?
	# TODO Keeping the else pass for now to remind me something might need to go here.









