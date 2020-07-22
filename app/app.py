import streamlit as st
import pandas as pd
import numpy as np
import sys
from collections import defaultdict


sys.path.append("../../oats")
from oats.utils.utils import save_to_pickle, load_from_pickle, merge_list_dicts, flatten, to_hms
from oats.utils.utils import function_wrapper_with_duration, remove_duplicates_retain_order
from oats.biology.dataset import Dataset



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
# Plant Phenotype Descriptions Dataset
Some description that goes under this. This web application is for doing x, y, and z. It was developed by Ian Braun
and also whatever and whover okay the options are shown below. Some description that goes under this. This web 
application is for doing x, y, and z. It was developed by Ian Braun
and also whatever and whover okay the options are shown below. [click here](github.com/irbraun/phenologs-with-oats)

'''




@st.cache(allow_output_mutation=True)
def read_in_oats_data(path):
	dataset = Dataset(path)
	dataset.filter_has_description()

	# Should eventually also return the loaded paired distances objects and preprocessing stuff, etc.
	return(dataset)













def truncate_string(text, char_limit):
	truncated_text = text[:char_limit]
	if len(text)>char_limit:
		truncated_text = "{}...".format(truncated_text)
	return(truncated_text)






@st.cache
def read_in_dataset(path):
	
	# Read in the dataset that was used for this analysis.
	df = pd.read_csv(path, sep="\t")
	
	# Create a set of new columns that are specific to how the information should be displayed in the app.
	df["Gene or Protein"] = df["gene_names"].map(lambda x: x.split("|")[0])
	df["Phenotype Description"] = df["description"]
	df["Phenotype Description (Truncated)"] = df["description"].map(lambda x: truncate_string(x, 500))

	return(df)






def gene_name_search(dataset, gene_name):


	# TODO have to make the dataset class support this, it doesn't yet.
	# TODO add the get_id() thing. make sure the species thing works().
	# Not currently doing any fuzzy matching, it's got to be in the dataset.
	# However, case is ignored when looking for a matching gene name or identifier.

	species_to_gene_id_list = defaultdict(list)
	for species in dataset.get_species():
		gene_ids = dataset.get_species_to_name_to_ids_dictionary(include_synonyms=True)[species][gene_name]
		for gene_id in gene_ids:
			species_to_gene_id_list[species].append(gene_id)
	return(species_to_gene_id_list)







def new_phenotype_description_search(distances, text, preprocessing):
	return(0)
	# TODO process the text with the provided preprocessing approach? figure out how to do this.
	# TODO Request the best matches from the distances object for that text.
	# TODO Return those list of IDs, so that the dataframe can be ordered that way.




def old_phenotype_description_search(distances, gene_id):
	return(0)
	# TODO get the IDs that are the closet to that ID bases on descripiton similarity as given
	# in that distances object. Return them as a list so that the the table can be re-ordered.




	















# in the dataset class, make a function that takes species name and gene name and returns a list of IDs.
# It should return a list, because multiple genes might take the same synonym or something like that.
# It shouldn't be limited onlyt to the primary gene names.

# Adjust the preprocessing notebooks to try a little harder to make the first gene name a good one.
# Use fuzzy matching to the accepted type of identifer like At3G* or something like that, or regex.






# Read in the dataframe with the full set of all columns.
#df = read_in_dataset(path="/Users/irbraun/plant-data/genes_texts_annots.tsv")



# Reading in the dataset object.



dataset = read_in_oats_data("/Users/irbraun/Desktop/test.csv")

# These mappings are necessary if the internal strings used for species and how they should be displayed are different.
internal_species_strings = ["ath", "zma", "sly", "gmx", "osa", "mtr"]
display_species_strings = ["Arabidopsis", "Maize", "Tomato", "Soybean", "Rice", "Medicago"]
to_species_display_name = {i:d for i,d in zip(internal_species_strings,display_species_strings)}





st.markdown("### Search by Phenotype Description")
s ="The NLP method selected in the sidebar will be used to find the phenotype descriptions that most closely resemble the provided text."
search_string = st.text_input(label=s)


st.markdown("### Search by Keywords or Phrases")
st.markdown("Only phenotype descriptions that contain the provided words or phrases (separated by commas if more than one) will be returned.")
search_kws = st.text_input(label="Enter text here")





write_button = st.button(label="Download All Data")
write_button = st.button(label="Download This Data")





















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










# Displaying the search results based on which genes or proteins were matching in the dataset.
st.sidebar.markdown("### Search by Gene Name or Identfier")
gene_search_string = st.sidebar.text_input(label="Enter a name or identifier for a gene or protein")











# Do the actual processing of the search against the full dataset here.
gene_matches = gene_name_search(dataset=dataset, gene_name=gene_search_string)

# Search text was entered and the search was processed but the list of relevant IDs found is empty.
if gene_search_string and len(gene_matches)==0:
	st.sidebar.markdown("No genes were found that match '{}'".format(gene_search_string))

# Search text was entered and the search was processed and atleast one matching ID was found.
gene_buttons_dict = {}
if gene_search_string and len(gene_matches)>0:
	st.sidebar.markdown("Click on a gene name below to organize the dataset by that gene.")
	for species,id_list in gene_matches.items():
		st.sidebar.markdown("{} matches".format(to_species_display_name[species]))
		for i in id_list:
			primary_gene_name = dataset.get_gene_dictionary()[i].primary_name
			other_names = dataset.get_gene_dictionary()[i].all_names
			gene_buttons_dict[i] = st.sidebar.button(label=primary_gene_name)
			if synonyms:
				st.sidebar.markdown("({})".format(", ".join(other_names)))














# The dataframe as it should be formatted for displaying in the app, not all data is needed.
df = dataset.to_pandas()
df["Species"] = df["species"].map(to_species_display_name)
df = df[df["Species"].isin(species_list)]
df["Gene or Protein"] = df["gene_names"].map(lambda x: x.split("|")[0])
df["Phenotype Description"] = df["description"]
df["Phenotype Description (Truncated)"] = df["description"].map(lambda x: truncate_string(x, 800))
# Make sure the data is always sorted by the internal IDs, but make the displayed index start at 1.
df.sort_values(by="id", inplace=True)
df.index = np.arange(1, len(df)+1)





# Check each of the presented buttons for genes that matched the search and if selected, present information for that gene.
for i,gene_button in gene_buttons_dict.items():
	if gene_button:
		selected_gene_primary_name = dataset.get_gene_dictionary()[i].primary_name
		selected_gene_other_names = dataset.get_gene_dictionary()[i].all_names
		selected_gene_phenotype_description = dataset.get_description_dictionary()[i]
		st.markdown("---")
		st.markdown("## Search Results")
		st.markdown("**Gene Selected:** {}".format(selected_gene_primary_name))
		st.markdown("**Possible Synonyms:** {}".format(", ".join(selected_gene_other_names)))
		st.markdown("**Phenotype Description(s):** {}".format(selected_gene_phenotype_description))
		st.markdown("### Genes with Similar Phenotypes")
		st.markdown("The dataset of plant genes below is now sorted by similarity to **{}**, as determined by the selected phenotype similarity approach in the sidebar.".format(selected_gene_primary_name))


		# TODO get the correct order of the other IDs based on their similarity to this selected gene.
		# TODO sort the dataframe before showing it below.







# Display the rest of the full dataset.
if truncate:
	st.table(data=df[["Species", "Gene or Protein", "Phenotype Description (Truncated)"]])
else:
	st.table(data=df[["Species", "Gene or Protein", "Phenotype Description"]])


if write_button:
	df.to_csv("data.tsv", index=False, sep="\t")




