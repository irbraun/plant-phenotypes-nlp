import streamlit as st
import pandas as pd
import numpy as np
import sys
import itertools
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import defaultdict
from string import punctuation
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import strip_non_alphanum, stem_text, preprocess_string


sys.path.append("../../oats")
from oats.utils.utils import save_to_pickle, load_from_pickle, merge_list_dicts, flatten, to_hms
from oats.utils.utils import function_wrapper_with_duration, remove_duplicates_retain_order
from oats.biology.dataset import Dataset
from oats.biology.groupings import Groupings
from oats.biology.relationships import ProteinInteractions, AnyInteractions
from oats.annotation.ontology import Ontology
from oats.annotation.annotation import annotate_using_noble_coder
from oats.distances import pairwise as pw
from oats.distances.edgelists import merge_edgelists, make_undirected, remove_self_loops, subset_with_ids
from oats.nlp.vocabulary import get_overrepresented_tokens, get_vocab_from_tokens
from oats.nlp.vocabulary import reduce_vocab_connected_components, reduce_vocab_linares_pontes
from oats.nlp.preprocess import concatenate_with_bar_delim



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




# Markdown for introducing the app and linking to other relevant resources like the project Github page.

'''
# Phenotype Description Search
Some description that goes under this. This web application is for doing x, y, and z. It was developed by Ian Braun
and also whatever and whover okay the options are shown below. Some description that goes under this. This web 
application is for doing x, y, and z. It was developed by Ian Braun
and also whatever and whover okay the options are shown below. [click here](github.com/irbraun/phenologs-with-oats)

'''






# TODO
# this script doesnt actually use the graph.arry or graph.df objects in the PairwiseDistance objects.
# hackily set them to = None before saving as a pickle in order to save a ton of space when loading this app.



# TODO
# Adjust the preprocessing notebooks to try a little harder to make the first gene name a good one.
# Use fuzzy matching to the accepted type of identifer like At3G* or something like that, or regex.










# Note that the output object(s) and dictionaries shouldn't actually be modified in this script.
# But streamlit was detecting a change in the outputs of this function? So the allow output mutation
# argument is necessary in order for this cached function to not be run again on fresh.
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def read_in_files(dataset_path, approach_names_and_data, approach_mapping_files):
	"""Summary
	
	Args:
	    dataset_path (TYPE): Description
	    approach_names_and_data (TYPE): Description
	    approach_mapping_files (TYPE): Description
	
	Returns:
	    TYPE: Description
	"""

	with st.spinner("Reading very large files, this might take a few minutes."):
		dataset = Dataset(dataset_path)
		dataset.filter_has_description()
		approach_to_object = {k:load_from_pickle(v["path"]) for k,v in approach_names_and_data.items()}
		mapping_key_to_mapping_dict = {k:load_from_pickle(v) for k,v in approach_mapping_files.items()}
		approach_to_mapping = {k:mapping_key_to_mapping_dict[v["mapping"]] for k,v in approach_names_and_data.items()}
		return(dataset, approach_to_object, approach_to_mapping)








def truncate_string(text, char_limit):
	truncated_text = text[:char_limit]
	if len(text)>char_limit:
		# Make extra room for the ... and then add it.
		truncated_text = text[:char_limit-3]
		truncated_text = "{}...".format(truncated_text)
	return(truncated_text)





def gene_name_search(dataset, gene_name):
	gene_name = gene_name.lower().strip()
	species_to_gene_id_list = defaultdict(list)
	for species in dataset.get_species():
		gene_ids = dataset.get_species_to_name_to_ids_dictionary(include_synonyms=True, lowercase=True)[species][gene_name]
		for gene_id in gene_ids:
			species_to_gene_id_list[species].append(gene_id)
	return(species_to_gene_id_list)






	



def keyword_search(ids_to_texts, keywords):

	# TODO shouldn't be checking if kw is in text string, should be checking if kw is in token list.
	# TODO that was 'larg' isn't found in a description that contains the word large etc.

	# TODO add the option to do stemming / case normalization prior to doing this matching too.
	# TODO should be done in the cached function and just returned as dictionaries of IDs --> token lists etc.

	id_to_found_keywords = {i:[kw for kw in keywords if kw in word_tokenize(text)] for i,text in ids_to_texts.items()}
	id_to_num_found_keywords = {i:len(kw_list) for i,kw_list in id_to_found_keywords.items()}
	return(id_to_found_keywords, id_to_num_found_keywords)





def description_search(text, graph, tokenization_function, preprocessing_function):


	# TODO Whether or not to split by sentences and then how to preprocess should be based on approach, not the same.
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
	# Some of this formatting is a work-around for not having alot of control over column widths and text-wrapping in the streamlit table.
	width = 30
	num_lines_limit = 5
	gene_id_to_result_string = {}
	for gene_id, distances in gene_id_to_distances.items():
		lines_with_dist_list = []
		for s,d in zip(preprocessed_sentence_tokens,distances):
			parsed_string_truncated = truncate_string(s, width)
			parsed_string_truncated = parsed_string_truncated + "."*max(0,width-len(parsed_string_truncated))
			line = "{}({:.2f})\n\n".format(parsed_string_truncated, d)
			lines_with_dist_list.append((line,d))
		# Sort that list of lines by distance, because we only want to show the best matches if the description is very long.
		lines_with_dist_list = sorted(lines_with_dist_list, key=lambda x: x[1])
		line_string = "".join([x[0] for x in lines_with_dist_list][:num_lines_limit])
		gene_id_to_result_string[gene_id] = line_string




	# For now, just get a mapping between gene IDs and their minimum distance to any of those parsed strings.
	# Maybe this should take more than just the minimum of them into account for this application?
	gene_id_to_min_distance = {}
	for gene_id,distances in gene_id_to_distances.items():
		gene_id_to_min_distance[gene_id] = min(distances)


	return(gene_id_to_result_string, gene_id_to_min_distance)

















# LOOK AT THIS.
# Dangerous part. These have to exactly match how the text is treated within the notebook that generates the pairwise distances.
# There's no explicit check in the code that makes sure the processing is identical between those two locations.
# In the future this should probably be changed so both sets of code are calling a resources that knows how to do these things.
sentence_tokenize = lambda text: sent_tokenize(text)
as_one_token = lambda text: [text]
identify_function = lambda text: text
simple_preprocessing = lambda text: " ".join(simple_preprocess(text))
full_preprocessing = lambda text: " ".join(preprocess_string(text))










# What are the different fields for each approach in this nested dictionary?
# Maybe this should all be moved to separate json file that references something to find those lambdas by key.
# path: The path to the pickle for loading the oats object associated with this approach.
# mapping: A key that allows for retrieving the correct mapping between IDs used here and those used internally by each approach.
# tokenization_function: A function for how text should be tokenized in order to be compatible with this approach.
# preprocessing_function: A function for how text should be preprocessed in order to be compatible with this approach.
approach_names_and_data = {
	"n-grams":{
		"path":"/Users/irbraun/Desktop/testp/n.pickle", 
		"mapping":"whole_texts",
		"tokenization_function":as_one_token,
		"preprocessing_fucntion":full_preprocessing,
		},
	"n-grams-tok":{
		"path":"/Users/irbraun/Desktop/testp/ntok.pickle", 
		"mapping":"sent_tokens",
		"tokenization_function":sentence_tokenize,
		"preprocessing_fucntion":full_preprocessing,
		},
	"word2vec-tok":{
		"path":"/Users/irbraun/Desktop/testp/wtok.pickle", 
		"mapping":"sent_tokens",
		"tokenization_function":sentence_tokenize,
		"preprocessing_fucntion":identify_function,
		},
	# "doc2vec":{
	# 	"path":"/Users/irbraun/Desktop/testp/d.pickle", 
	# 	"mapping":"whole_texts",
	# 	"tokenization_function":as_one_token,
	# 	"preprocessing_fucntion":identify_function,
	# 	},
	}




approach_mapping_files = {
	"whole_texts":"/Users/irbraun/Desktop/testp/gene_id_to_unique_ids_whole_texts.pickle",
	"sent_tokens":"/Users/irbraun/Desktop/testp/gene_id_to_unique_ids_sent_tokens.pickle",
	}




dataset_path = "/Users/irbraun/Desktop/test.csv"



# The first dictionary created here maps the name of an apporach, like "n-grams" to an oats.distances object.
# The second dictionary created here maps the name of an approach like "n-grams" to another dictionary that
# maps the IDs that in this dataset to the IDs that are used internally by that corresponding distances object.
# This is necessary because those internal IDs represent compressed or altered versions of the text and could 
# refer to more than one text instance in the actual dataset that is seen here.
dataset, approach_to_object, approach_to_mapping = read_in_files(dataset_path, approach_names_and_data, approach_mapping_files)











# These mappings are necessary if the internal strings used for species and how they should be displayed are different.
internal_species_strings = ["ath", "zma", "sly", "gmx", "osa", "mtr"]
display_species_strings = ["Arabidopsis", "Maize", "Tomato", "Soybean", "Rice", "Medicago"]
to_species_display_name = {i:d for i,d in zip(internal_species_strings,display_species_strings)}





st.markdown("### Search Phenotype Descriptions with NLP")
st.markdown("Find phenotype descriptions that are most similar to text of any length, using the semantic similarity approach selected in the sidebar.")
search_string = st.text_input(label="Enter a phenotype description here")


st.markdown("### Simple Keyword or Phrase Search")
st.markdown("Find phenotype descriptions that contain specific words or phrases. Separate words or phrases by commas if searching for more than one.")
search_kws = st.text_input(label="Enter keywords or phrases here")

















############# The Sidebar ###############





# Presenting the general options for how the data is displayed.
st.sidebar.markdown("### General Options")
truncate = st.sidebar.checkbox(label="Truncate long phenotypes", value=True)
synonyms = st.sidebar.checkbox(label="Show possible gene synonyms", value=False)

# Presenting the options for filtering by species.
st.sidebar.markdown("### Filtering by Species")
species_display_names = [to_species_display_name[x] for x in dataset.get_species()]
species_list = st.sidebar.multiselect(label="Filter to only include certain species", options=species_display_names)
if len(species_list) == 0:
	species_list = species_display_names






# Present information and search options for finding a particular gene or protein in the sidebar.
st.sidebar.markdown("### Selecting an Approach")
st.sidebar.markdown("Use a particular type of NLP approach to find genes that are related to some phenotype description of interest.")
approach = st.sidebar.selectbox(label="Pick one", options=list(approach_to_object.keys()), index=0)

# Variables that get set according to which approach was selected.
graph = approach_to_object[approach]
gene_id_to_graph_ids = approach_to_mapping[approach]







# Displaying the search results based on which genes or proteins were matching in the dataset.
st.sidebar.markdown("### Search by Gene Name or Identfier")
gene_search_string = st.sidebar.text_input(label="Enter a name or identifier for a gene or protein")



# TODO add download buttons.














############## Default view, presenting the entire dataset ################




# The dataframe as it should be formatted for displaying in the app, not all data is needed.
df = dataset.to_pandas()
df["Species"] = df["species"].map(to_species_display_name)
df = df[df["Species"].isin(species_list)]
df["Gene or Protein"] = df["gene_names"].map(lambda x: x.split("|")[0])
df["Phenotype Description"] = df["description"]
df["Phenotype Description (Truncated)"] = df["description"].map(lambda x: truncate_string(x, 800))









############### Listening for something to typed and entered into any of the search boxes #############







# Handle what should happen if a search for a gene name is performed.
if gene_search_string:

	# Do the actual processing of the search against the full dataset here.
	gene_matches = gene_name_search(dataset=dataset, gene_name=gene_search_string)

	# Search text was entered and the search was processed but the list of relevant IDs found is empty.
	if len(gene_matches)==0:
		st.sidebar.markdown("No genes were found that match '{}'".format(gene_search_string))

	# Search text was entered and the search was processed and atleast one matching ID was found.
	gene_buttons_dict = {}
	if len(gene_matches)>0:
		st.sidebar.markdown("Click on a gene name below to organize the dataset by that gene.")
		for species,id_list in gene_matches.items():
			st.sidebar.markdown("{} matches".format(to_species_display_name[species]))
			for i in id_list:
				primary_gene_name = dataset.get_gene_dictionary()[i].primary_name
				other_names = dataset.get_gene_dictionary()[i].all_names
				gene_buttons_dict[i] = st.sidebar.button(label=primary_gene_name)
				if synonyms:
					st.sidebar.markdown("({})".format(", ".join(other_names)))













	# Handle what should happen if any of the previously presented gene buttons was clicked.
	# Has to be a loop because we need to check all the presented buttons, might be more than one.
	for i,gene_button in gene_buttons_dict.items():
		if gene_button:
			
			# Get information about which gene from the dataset was selected.
			selected_gene_primary_name = dataset.get_gene_dictionary()[i].primary_name
			selected_gene_other_names = dataset.get_gene_dictionary()[i].all_names
			selected_gene_phenotype_description = dataset.get_description_dictionary()[i]
			
			# Describe what the results of the search are and what they mean in markdown.
			st.markdown("---")
			st.markdown("## Search Results (Gene)")
			st.markdown("**Gene Selected:** {}".format(selected_gene_primary_name))
			st.markdown("**Possible Synonym(s):** {}".format(", ".join(selected_gene_other_names)))
			st.markdown("**Phenotype Description(s):** {}".format(selected_gene_phenotype_description))
			st.markdown("### Genes with Similar Phenotypes")
			st.markdown("The dataset of plant genes below is now sorted by similarity to **{}**, as determined by the selected phenotype similarity approach in the sidebar.".format(selected_gene_primary_name))

			
			# Perform a secondary search using this gene's phenotype description to organize the rest of the data.
			search_string = dataset.get_description_dictionary()[i]
			f_tokenizing = approach_names_and_data[approach]["tokenization_function"]
			f_preprocessing = approach_names_and_data[approach]["preprocessing_fucntion"]
			gene_id_to_result_string, gene_id_to_min_distance =  description_search(search_string, graph, f_tokenizing, f_preprocessing)

			df["ResultResultResultResultResultResult"] = df["id"].map(gene_id_to_result_string)
			df["distance"] = df["id"].map(gene_id_to_min_distance)
			df.sort_values(by=["distance","id"], ascending=[True,True], inplace=True)


			# Display the sorted and filtered dataset as a table with the relevant columns.
			df.index = np.arange(1, len(df)+1)
			if truncate:
				st.table(data=df[["ResultResultResultResultResultResult", "Species", "Gene or Protein", "Phenotype Description (Truncated)"]])
			else:
				st.table(data=df[["ResultResultResultResultResultResult", "Species", "Gene or Protein", "Phenotype Description"]])







# A search using the keyword or phrase text box.
if search_kws:

	# Do the processing of the search and add necessary columns to the dataframe.
	keywords = search_kws.strip().strip(punctuation).split(",")
	keywords = [kw.strip() for kw in keywords]
	id_to_found_keywords, id_to_num_found_keywords = keyword_search(dataset.get_description_dictionary(), keywords)
	df["num_found"] = df["id"].map(id_to_num_found_keywords)
	df = df[df["num_found"]>0]
	df["Keywords"] = df["id"].map(lambda x: ", ".join(id_to_found_keywords[x]))
	df.sort_values(by=["num_found","id"], ascending=[False,True], inplace=True)


	# Describe what the results of the search are and what they mean in markdown.
	st.markdown("---")
	st.markdown("## Search Results (Keyword)")
	keywords_str = ", ".join([kw for kw in keywords if len(kw.split())==1])
	phrases_str = ", ".join([kw for kw in keywords if len(kw.split())>1])
	st.markdown("**Keyword(s)**: {}".format(keywords_str))
	st.markdown("**Phrase(s)**: {}".format(phrases_str))
	st.markdown("### Genes with Matching Phenotype Descriptions")
	st.markdown("The dataset of plant genes below shows only genes with phenotype description that included one or more of the searched keywords or phrases.")



	# Display the sorted and filtered dataset as a table with the relevant columns.
	df.index = np.arange(1, len(df)+1)
	if truncate:
		st.table(data=df[["Keywords", "Species", "Gene or Protein", "Phenotype Description (Truncated)"]])
	else:
		st.table(data=df[["Keywords", "Species", "Gene or Protein", "Phenotype Description"]])







# A search using the phenotype description text box.
elif search_string:

	# Do the processing of the search and add necessary columns to the dataframe to be shown.
	f_tokenizing = approach_names_and_data[approach]["tokenization_function"]
	f_preprocessing = approach_names_and_data[approach]["preprocessing_fucntion"]
	gene_id_to_result_string, gene_id_to_min_distance =  description_search(search_string, graph, f_tokenizing, f_preprocessing)
	df["ResultResultResultResultResultResult"] = df["id"].map(gene_id_to_result_string)
	df["distance"] = df["id"].map(gene_id_to_min_distance)
	df.sort_values(by=["distance","id"], ascending=[True,True], inplace=True)



	# Describe what the results of the search are and what they mean in markdown.
	st.markdown("---")
	st.markdown("## Search Results (NLP)")
	st.markdown("### Genes with Matching Phenotype Descriptions")
	st.markdown("Genes with phenotypes that described most similarity to '{}'".format(search_string))



	# Display the sorted and filtered dataset as a table with the relevant columns.
	df.index = np.arange(1, len(df)+1)
	if truncate:
		st.table(data=df[["ResultResultResultResultResultResult", "Species", "Gene or Protein", "Phenotype Description (Truncated)"]])
	else:
		st.table(data=df[["ResultResultResultResultResultResult", "Species", "Gene or Protein", "Phenotype Description"]])




# Nothing was selected, default to just showing the whole dataset.
else:


	# Describe what is being shown.
	st.markdown("---")
	st.markdown("## Viewing the Entire Dataset")
	st.markdown("Search for a gene, keyword, phrase, or phenotype within this dataset.".format(search_string))


	# Display the full dataframe before any search is performed.
	df.sort_values(by="id", inplace=True)
	df.index = np.arange(1, len(df)+1)
	if truncate:
		st.table(data=df[["Species", "Gene or Protein", "Phenotype Description (Truncated)"]])
	else:
		st.table(data=df[["Species", "Gene or Protein", "Phenotype Description"]])













