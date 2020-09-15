#!/usr/bin/env python
# coding: utf-8

# # Looking at the Dataset
# The purpose of this notebook is to look closer at the dataset of genes, natural language descriptions, and ontology term annotations that are used in this work. As included in the preprocessing notebooks, these data are drawn from files from either publications supplements like Oellrich, Walls et al. (2015) or model species databases such as TAIR, MaizeGDB, and SGN. The datasets are already loaded and merged using classes available through the oats package.

# In[1]:


import datetime
import nltk
import pandas as pd
import numpy as np
import time
import math
import sys
import gensim
import os
import random
import warnings
from collections import defaultdict
from nltk.corpus import brown
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from gensim.parsing.preprocessing import strip_non_alphanum, stem_text, preprocess_string, remove_stopwords
from gensim.utils import simple_preprocess
from nltk.stem import WordNetLemmatizer
from statsmodels.sandbox.stats.multicomp import multipletests

sys.path.append("../../oats")
sys.path.append("../../oats")
from oats.utils.utils import save_to_pickle, load_from_pickle, flatten, to_hms
from oats.utils.utils import function_wrapper_with_duration, remove_duplicates_retain_order
from oats.biology.dataset import Dataset
from oats.biology.groupings import Groupings
from oats.biology.relationships import ProteinInteractions, AnyInteractions
from oats.annotation.ontology import Ontology
from oats.annotation.annotation import annotate_using_noble_coder, term_enrichment
from oats.distances import pairwise as pw
from oats.nlp.vocabulary import get_overrepresented_tokens, get_vocab_from_tokens
from oats.nlp.vocabulary import reduce_vocab_connected_components, reduce_vocab_linares_pontes, token_enrichment

warnings.simplefilter('ignore')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
nltk.download('punkt', quiet=True)
nltk.download('brown', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)


# In[2]:


# Paths to the files that are used for this notebook.
plant_dataset_path = "../../plant-data/genes_texts_annots.csv"

# Paths to files with mappings to groups.
kegg_pathways_path = "../../plant-data/reshaped_data/kegg_pathways.csv" 
plantcyc_pathways_path = "../../plant-data/reshaped_data/plantcyc_pathways.csv" 
lloyd_meinke_subsets_path = "../../plant-data/reshaped_data/lloyd_meinke_subsets.csv" 
lloyd_meinke_classes_path = "../../plant-data/reshaped_data/lloyd_meinke_classes.csv" 

# Paths to files that map group identifers to longer group names.
kegg_pathways_names_path = "../../plant-data/reshaped_data/kegg_pathways_name_map.csv"
plantcyc_pathways_names_path = "../../plant-data/reshaped_data/plantcyc_pathways_name_map.csv"
lloyd_meinke_subsets_names_path = "../../plant-data/reshaped_data/lloyd_meinke_subsets_name_map.csv"
lloyd_meinke_classes_names_path = "../../plant-data/reshaped_data/lloyd_meinke_classes_name_map.csv"

# Path to file with plant ortholog mappings.
ortholog_file_path = "../../plant-data/databases/panther/PlantGenomeOrthologs_IRB_Modified.txt"


# In[3]:


# Create and name an output directory according to when the notebooks was run.
OUTPUT_NAME = "composition"
OUTPUT_DIR = os.path.join("../outputs","{}_{}_{}".format(OUTPUT_NAME,datetime.datetime.now().strftime('%m_%d_%Y_h%Hm%Ms%S'),random.randrange(1000,9999)))
os.mkdir(OUTPUT_DIR)


# In[4]:


# Reading in and describing the dataset of plant genes.
plant_dataset = Dataset(plant_dataset_path)
plant_dataset.filter_has_description()
plant_dataset.describe()


# ### What's there for each species?
# The previously loaded dataset contains all of the genes that across six plant species that have natural language description data for phenotype(s) related to that gene. Each gene can have multiple descriptions annotated to it, which were combined or concatenated when the datasets from multiple sources were merged in creating the pickled datasets. Arabidopsis has the highest number of genes that satisfy this criteria, followed by maize, and then followed by the other four species which have a relatively low number of genes that satisfy this criteria, atleast given the sources used for this work. Note that the number of unique descriptions is lower than the number of genes in call cases, because multiple genes can have the same phenotype description associated with them.

# In[5]:


data = plant_dataset

wnl = WordNetLemmatizer()
lemmatize_doc = lambda d: [wnl.lemmatize(x) for x in simple_preprocess(d)]

dists = defaultdict(list)

sent_lists = {}
token_lists = {}
stems_lists = {}
lemma_lists = {}


# For each individual species.
for species in data.get_species():
    df = data.to_pandas()
    subset = df[df["species"]==species]
    sentences = [sent_tokenize(d) for d in subset["descriptions"].values]
    descriptions_not_stemmed = [simple_preprocess(d) for d in subset["descriptions"].values]
    descriptions_stemmed = [preprocess_string(d) for d in subset["descriptions"].values]
    descriptions_lemmatized = [lemmatize_doc(d) for d in subset["descriptions"].values]
    sent_lists[species] = flatten(sentences)
    token_lists[species] = flatten(descriptions_not_stemmed)
    stems_lists[species] = flatten(descriptions_stemmed)    
    lemma_lists[species] = flatten(descriptions_lemmatized)
    
    # What about the distributions of words per gene and sentences per gene?
    dists["species"].extend([species]*subset.shape[0])
    dists["num_words"].extend([len(word_tokenize(x)) for x in subset["descriptions"].values])
    dists["num_sents"].extend([len(sent_tokenize(x)) for x in subset["descriptions"].values])
    
# For the entire dataset including all of the species.
df = data.to_pandas()
subset = df
sentences = [sent_tokenize(d) for d in subset["descriptions"].values]
descriptions_not_stemmed = [simple_preprocess(d) for d in subset["descriptions"].values]
descriptions_stemmed = [preprocess_string(d) for d in subset["descriptions"].values]
descriptions_lemmatized = [lemmatize_doc(d) for d in subset["descriptions"].values]
sent_lists["total"] = flatten(sentences)
token_lists["total"] = flatten(descriptions_not_stemmed)
stems_lists["total"] = flatten(descriptions_stemmed)    
lemma_lists["total"] = flatten(descriptions_lemmatized)

# What about lemmas that are uniquely used for a particular species?
lemma_sets_unique_to_species = {}
for species in data.get_species():
    other_species = [s for s in data.get_species() if s != species]
    lemmas_used_in_other_species = set(flatten([lemma_lists[s] for s in other_species]))
    unique_lemmas = set(lemma_lists[species]).difference(lemmas_used_in_other_species)
    lemma_sets_unique_to_species[species] = unique_lemmas
lemma_sets_unique_to_species["total"] = flatten([list(s) for s in lemma_sets_unique_to_species.values()])

    
# Create a dataframe to contain the summarizing information about this dataset, and sort it by number of genes.
# Unique gene identifiers is just the total number of genes, this column name should be changed in the class...
df = data.describe() 
condition = (df.species=="total")
excluded = df[condition]
included = df[~condition]
df_sorted = included.sort_values(by="unique_gene_identifiers", ascending=False)
df = pd.concat([df_sorted,excluded])

# Add columns summarizing information about the text descriptions in the dataset.
df["total_sents"] = df["species"].map(lambda x: len(sent_lists[x]))
df["total_words"] = df["species"].map(lambda x: len(token_lists[x]))
df["unique_words"] = df["species"].map(lambda x: len(set(token_lists[x])))
df["unique_stems"] = df["species"].map(lambda x: len(set(stems_lists[x])))
df["total_lemmas"] = df["species"].map(lambda x: len(lemma_lists[x]))
df["unique_lemmas"] = df["species"].map(lambda x: len(set(lemma_lists[x])))
df["unique_lemmas_to_species"] = df["species"].map(lambda x: len(lemma_sets_unique_to_species[x]))
df


# In[6]:


text_distributions = pd.DataFrame(dists)
text_distributions.to_csv(os.path.join(OUTPUT_DIR, "word_sent_distributions.csv"), index=False)
text_distributions.head(20)


# ### What about the ontology term annotations for each species?

# In[7]:


# How many of the genes in this dataset for each species are mapped to atleast one term from a given ontology?
num_mapped_go = {}
num_mapped_po = {}
for species in data.get_species():
    d = data.to_pandas()
    subset = d[d["species"]==species]    
    num_mapped_po[species] = len([t for t in subset["annotations"].values if "PO" in t])
    num_mapped_go[species] = len([t for t in subset["annotations"].values if "GO" in t])
num_mapped_go["total"] = sum(list(num_mapped_go.values()))   
num_mapped_po["total"] = sum(list(num_mapped_po.values()))
df["go"] = df["species"].map(lambda x: num_mapped_go[x])
df["po"] = df["species"].map(lambda x: num_mapped_po[x])
df


# ### What about the biologically relevant groups like biochemical pathways and phenotypes?

# In[8]:


# What are the groupings that we're interested in mapping to? Uses the paths defined at the top of the notebook.
groupings_dict = {
    "kegg":(kegg_pathways_path, kegg_pathways_names_path),
    "plantcyc":(plantcyc_pathways_path, plantcyc_pathways_names_path),
    "lloyd_meinke":(lloyd_meinke_subsets_path, lloyd_meinke_subsets_names_path)
}


for name,(filename,mapfile) in groupings_dict.items():
    groups = Groupings(filename, {row.group_id:row.group_name for row in pd.read_csv(mapfile).itertuples()})
    id_to_group_ids, group_id_to_ids = groups.get_groupings_for_dataset(data)
    group_mapped_ids = [k for (k,v) in id_to_group_ids.items() if len(v)>0]
    species_dict = data.get_species_dictionary()
    num_mapped = {}
    for species in data.get_species():
        num_mapped[species] = len([x for x in group_mapped_ids if species_dict[x]==species])
    num_mapped["total"] = sum(list(num_mapped.values()))    
    df[name] = df["species"].map(lambda x: num_mapped[x])  
df


# ### What about the other biologically relevant information like orthologous genes and protein interactions?

# In[9]:


# PantherDB for plant orthologs.
ortholog_edgelist = AnyInteractions(data.get_name_to_id_dictionary(), ortholog_file_path)
species_dict = data.get_species_dictionary()
num_mapped = {}
for species in data.get_species():
    num_mapped[species] = len([x for x in ortholog_edgelist.ids if species_dict[x]==species])
num_mapped["total"] = sum(list(num_mapped.values()))
df["panther"] = df["species"].map(lambda x: num_mapped[x])    
df


# In[10]:


# STRING DB for protein-protein interactions.
naming_file = "../../plant-data/databases/string/all_organisms.name_2_string.tsv"
interaction_files = [
    "../../plant-data/databases/string/3702.protein.links.detailed.v11.0.txt", # Arabidopsis
    "../../plant-data/databases/string/4577.protein.links.detailed.v11.0.txt", # Maize
    "../../plant-data/databases/string/4530.protein.links.detailed.v11.0.txt", # Tomato 
    "../../plant-data/databases/string/4081.protein.links.detailed.v11.0.txt", # Medicago
    "../../plant-data/databases/string/3880.protein.links.detailed.v11.0.txt", # Rice 
    "../../plant-data/databases/string/3847.protein.links.detailed.v11.0.txt", # Soybean
    "../../plant-data/databases/string/9606.protein.links.detailed.v11.0.txt", # Human
]
genes = data.get_gene_dictionary()
string_data = ProteinInteractions(genes, naming_file, *interaction_files)
species_dict = data.get_species_dictionary()
num_mapped = {}
for species in data.get_species():
    num_mapped[species] = len([x for x in string_data.ids if species_dict[x]==species])
num_mapped["total"] = sum(list(num_mapped.values()))
df["stringdb"] = df["species"].map(lambda x: num_mapped[x])    
df


# In[11]:


# Write that dataframe with all the information about datast to a file.
df.to_csv(os.path.join(OUTPUT_DIR,"full_dataset_composition.csv"),index=False)


# ### How do the vocabularies used for different species compare?
# One of the things we are interested in is discovering or recovering phenotype similarity between different species in order to identify phenologs (phenotypes between species that share some underlying genetic cause). For this reason, we are interested in how the vocabularies used to describe phenotypes between different species vary, because this will impact how feasible it is to use a dataset like this to identify phenologs. Because the Arabidopsis and maize datasets are the largest in this case, we will compare the vocabularies used in describing the phenotypes associated with the genes from these species in this dataset.

# In[12]:


# Using lemmas as the vocabulary components.
vocabs = {s:set(lemma_list) for s,lemma_list in lemma_lists.items()}
fdist_zma = FreqDist(lemma_lists["zma"])
fdist_ath = FreqDist(lemma_lists["ath"])

# Using word stems as the vocabulary components.
#vocabs = {s:set(stems_list) for s,stems_list in stems_lists.items()}
#fdist_zma = FreqDist(stems_lists["zma"])
#fdist_ath = FreqDist(stems_lists["ath"])

# Using tokens (full words) as the vocabulary components.
#vocabs = {s:set(token_list) for s,token_list in token_lists.items()}
#fdist_zma = FreqDist(token_lists["zma"])
#fdist_ath = FreqDist(token_lists["ath"])

union_vocab = vocabs["zma"].union(vocabs["ath"])
table = pd.DataFrame({"token":list(union_vocab)})
stops = set(stopwords.words('english'))
table = table[~table.token.isin(stops)]
table["part_of_speech"] = table["token"].map(lambda x: nltk.pos_tag([x])[0][1][:2])
table["ath_freq"] = table["token"].map(lambda x: fdist_ath[x])
table["ath_rate"] = table["ath_freq"]*100/len(token_lists["ath"])
table["zma_freq"] = table["token"].map(lambda x: fdist_zma[x])
table["zma_rate"] = table["zma_freq"]*100/len(token_lists["zma"])
table["diff"] = table["ath_rate"]-table["zma_rate"]
table.to_csv(os.path.join(OUTPUT_DIR,"token_frequencies.csv"), index=False)
table.head(10)


# In[13]:


# What are the tokens more frequently used for Arabidopsis than maize descriptions in this dataset?
table.sort_values(by="diff", ascending=False, inplace=True)
table.head(30)


# In[14]:


# What are the tokens more frequently used for maize than Arabidopsis descriptions in this dataset?
table.sort_values(by="diff", ascending=True, inplace=True)
table.head(30)


# In[15]:


# Is the mean absolute value of the rate differences different between the different parts of speech?
table["abs_diff"] = abs(table["diff"])
pos_table = table.groupby("part_of_speech").mean()
pos_table.sort_values(by="abs_diff", inplace=True, ascending=False)
pos_table = pos_table[["abs_diff"]]
pos_table.reset_index()


# In[16]:


# Working on the Venn Diagram for this part, unused currently.
#print(table.shape)
#zma_only = table[table["ath_rate"]==0]
#ath_only = table[table["zma_rate"]==0]
#print(zma_only.shape)
#print(ath_only.shape)
#print(ath_only.shape[0]+zma_only.shape[0])
#ath_only.head(10)
# We need to create a mapping between stems and the words that were present for them.
# This is because what we want is the stems that are exclusive to a species.
# but then the words that are actually there for those stems, so that we can count their parts of speech.


# ### Looking at Term and Word Enrichment for Groups of Genes

# In[84]:


# Loading the dataset of phenotype descriptions and ontology annotations.
plant_dataset = Dataset(plant_dataset_path)
data = plant_dataset
data.filter_has_description()
#data.filter_has_annotation("GO")
data.filter_has_annotation("PO")
d = data.get_description_dictionary()
texts = {i:" ".join(simple_preprocess(t)) for i,t in d.items()}
len(texts)                              


# In[62]:


# Create ontology objects for all the biological ontologies being used.
go_pickle_path = "../ontologies/go.pickle"                                                                
po_pickle_path = "../ontologies/po.pickle"                                                             
pato_pickle_path = "../ontologies/pato.pickle"
pato = load_from_pickle(pato_pickle_path)
po = load_from_pickle(po_pickle_path)
go = load_from_pickle(go_pickle_path)


# In[85]:


curated_go_annotations = data.get_annotations_dictionary("GO")
curated_po_annotations = data.get_annotations_dictionary("PO")
print("done")


# In[77]:


# Which GO terms are used to annotate the most genes in this dataset?
term_id_to_ids = defaultdict(list)
for i,term_id_list in curated_go_annotations.items():
    for term_id in term_id_list:
        term_id_to_ids[term_id].append(i)
term_id_to_num_ids = {k:len(v) for k,v in term_id_to_ids.items()}
terms_df = pd.DataFrame(term_id_to_num_ids.items(), columns=["term_id", "freq"])

def get_term_name(ont,i):
    try:
        return(ont[i].name)
    except:
        return("")

terms_df["term_name"] = terms_df["term_id"].map(lambda x: get_term_name(go,x))
terms_df.sort_values(by="freq", ascending=False, inplace=True)
terms_df.head(20)


# In[78]:


# Make the group be ones that have that GO term anntation.
#go_term_id_of_interest = "GO:0009640"
#gene_ids_in_this_pathway = [k for k,v in curated_go_annotations.items() if go_term_id_of_interest in v]


# In[ ]:





# In[ ]:





# In[86]:


# Load the mappings from this dataset to PlantCyc information.
#pmn_pathways_filename = "../data/pickles/groupings_from_pmn_pathways.pickle"                        
#groups = load_from_pickle(pmn_pathways_filename)
#id_to_group_ids, group_id_to_ids = groups.get_groupings_for_dataset(data)


# Reading in the dataset of groupings for pathways in PlantCyc.
plantcyc_name_mapping = {row.group_id:row.group_name for row in pd.read_csv(plantcyc_pathways_names_path).itertuples()}
plantcyc_grouping = Groupings(plantcyc_pathways_path, plantcyc_name_mapping)
id_to_group_ids, group_id_to_ids = plantcyc_grouping.get_groupings_for_dataset(data)

# Look at which pathways are best represented in this dataset.
pathways_sorted = sorted(group_id_to_ids.items(), key=lambda item: len(item[1]), reverse=True)
pathways_sorted_lengths = [(i,len(l)) for (i,l) in pathways_sorted]
pathways_df = pd.DataFrame(pathways_sorted_lengths, columns=["pathway_id","num_genes"])
pathways_df["pathway_name"] = pathways_df["pathway_id"].map(lambda x: plantcyc_grouping.get_long_name(x))
pathways_df = pathways_df[["pathway_name","pathway_id","num_genes"]]
pathways_df.head(15)


# In[107]:


# For some example pathway to use.
#pathway_id = "PWY-361"
pathway_id = "PWY-581"
#pathway_id = "PWY-1121"
pathway_id = "PWY-695"
gene_ids_in_this_pathway = group_id_to_ids[pathway_id]
gene_ids_in_this_pathway


# In[98]:


wordcloud = defaultdict(list)


# In[108]:


results = term_enrichment(curated_po_annotations, gene_ids_in_this_pathway, po).head(20)
threshold = 0.05
results["p_value_adj"] = multipletests(results["p_value"].values, method='bonferroni')[1]
results["significant"] = results["p_value_adj"] < threshold
results = results.loc[results["significant"]==True]
results["info_content"] = results["term_id"].map(lambda x: po.ic(x))
results.sort_values(by="info_content", ascending=False, inplace=True)


# ns   P > 0.05
# *    P ≤ 0.05
# **   P ≤ 0.01
# ***  P ≤ 0.001
# **** P ≤ 0.0001

# This lambda won't work is passed a value greater than the minimum p-value for significance defined here.
significance_levels = {0.05:"*", 0.01:"**", 0.001:"***", 0.0001:"****"}
get_level = lambda x: significance_levels[min([level for level in significance_levels.keys() if x <= level])]

results["significance"] = results["p_value_adj"].map(get_level)
results


# In[109]:


for row in results.itertuples():
    wordcloud["Weight"].append(int(1/row.p_value_adj))
    wordcloud["Word"].append("{} ({})".format(row.term_id,row.term_label))


# In[102]:


results = term_enrichment(curated_go_annotations, gene_ids_in_this_pathway, go).head(20)

from statsmodels.sandbox.stats.multicomp import multipletests
threshold = 0.05
results["p_value_adj"] = multipletests(results["p_value"].values, method='bonferroni')[1]
results["significant"] = results["p_value_adj"] < threshold


results = results.loc[results["significant"]==True]

results["info_content"] = results["term_id"].map(lambda x: go.ic(x))
results.sort_values(by="info_content", ascending=False, inplace=True)


# This lambda won't work is passed a value greater than the minimum p-value for significance defined here.
significance_levels = {0.05:"*", 0.01:"**", 0.001:"***", 0.0001:"****"}
get_level = lambda x: significance_levels[min([level for level in significance_levels.keys() if x <= level])]

results["significance"] = results["p_value_adj"].map(get_level)
results


# In[110]:


results = token_enrichment(texts, gene_ids_in_this_pathway).head(20)


threshold = 0.05
results["p_value_adj"] = multipletests(results["p_value"].values, method='bonferroni')[1]
results["significant"] = results["p_value_adj"] < threshold
results = results.loc[results["significant"]==True]


# This lambda won't work if passed a value greater than the minimum p-value for significance defined here.
significance_levels = {0.05:"*", 0.01:"**", 0.001:"***", 0.0001:"****"}
get_level = lambda x: significance_levels[min([level for level in significance_levels.keys() if x <= level])]
results["significance"] = results["p_value_adj"].map(get_level)
results


# In[111]:


for row in results.itertuples():
    wordcloud["Weight"].append(int(1/row.p_value_adj))
    wordcloud["Word"].append(row.token)


# In[112]:


pd.DataFrame(wordcloud).to_csv(os.path.join(OUTPUT_DIR, "{}_word_cloud.csv".format(pathway_id)), index=False)
pd.DataFrame(wordcloud)

