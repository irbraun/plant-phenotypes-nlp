#!/usr/bin/env python
# coding: utf-8

# ### Making plots to illustrate the results of some queries with the web tool
# This notebook takes a file describing a list of genes with those genes species and identifying strings. It loads the dataset that is used by the search tool, finds out if those genes are in the dataset, and then subsets the set of genes to include just those. It then uses the descriptions of those genes to query the tool (using the script rather than the streamlit web app, but the results are identical), and keeps track of where other genes from the list fall in the resulting gene rankings. The output of the notebook is a summary of these results that specifies the mean and standard deviation of binned ranks for each search.

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
from itertools import product

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
dataset_path = "../../quoats/data/genes_texts_annots.csv"
dataset = Dataset(dataset_path, keep_ids=True)
dataset.describe()


# ### Anthocyanin biosynthesis genes

# Previously out of this list we had 10 of 16 maize genes in the dataset and 16 of 18 Aribidopsis genes. Now we have 13 of 16 maize genes and 18 of 18 Arabidopsis genes.

# In[3]:


mapping = dataset.get_species_to_name_to_ids_dictionary(include_synonyms=False, lowercase=True)
genes = pd.read_csv("anthocyanin_biosynthesis_genes.csv")
genes["id"] = genes.apply(lambda x: mapping[x["species_code"]].get(x["identifier"].strip().lower(),-1), axis=1)
genes[genes["id"]!=-1]["id"] = genes[genes["id"]!=-1]["id"].map(lambda x: x[0])
genes["in_current_dataset"] = genes["id"].map(lambda x: x!=-1)
genes

# Looking just at the ones that are in the current dataset.
genes = genes[genes["in_current_dataset"]]
genes["id"] = genes["id"].map(lambda x: x[0])
genes


# In[4]:


# Grabbing the texts dictionary from the dataset that we can use to grab the descriptions to query.
texts = dataset.get_description_dictionary()
texts[2617]


# In[5]:


# Prepare dictionaries to hold the resulting arrays.
resulting_bin_arrays = defaultdict(dict)
resulting_bin_arrays["zma"]["zma"] = []
resulting_bin_arrays["ath"]["ath"] = []
resulting_bin_arrays["zma"]["ath"] = []
resulting_bin_arrays["ath"]["zma"] = []
resulting_bin_arrays


# ### Searching within the same species

# In[7]:


# The searches within the same species.
rank_for_not_found = 100
bins =[0,11,21,31,41,51,rank_for_not_found]
bin_names = [10,20,30,40,50,rank_for_not_found]
assert len(bin_names) == len(bins)-1

ctr = 0
for gene in genes.itertuples():
    
    ctr = ctr+1
    limit = 50
    species = gene[1]
    species_code = gene[2]
    identifier = gene[3]
    gene_id = gene[5]
    text = texts[gene_id]
    
    # Because these are being passed as strings to the command line, quotes need to be removed now,
    # instead of waiting for them to be removed as a preprocessing step of the search strings in the streamlit script.
    text = text.replace("'","")
    text = text.replace('"','')
    
    path = "/Users/irbraun/phenologs-with-oats/quoats/outputs_within_anthocyanin/output_{}.tsv".format(ctr)
    os.chdir('/Users/irbraun/quoats')
    os.system("python main.py -s {} -t identifiers -q '{}:{}' -l {} -o {} -r 0.000 -a TFIDF".format(species,species,identifier,limit,path))
    time.sleep(4)
    df = pd.read_csv(path, sep='\t')
    df = df[["Rank","Internal ID"]]
    df = df.drop_duplicates()
    id_to_rank = dict(zip(df["Internal ID"].values,df["Rank"].values))
    assert rank_for_not_found > limit
    
    # For within the same species, get rid of the identical gene (always rank 1).
    ids_of_interest = [i for i in genes[genes["species"]==species]["id"].values if i != gene_id]
    ranks = [id_to_rank.get(i, rank_for_not_found) for i in ids_of_interest]
    resulting_bin_arrays[species_code][species_code].append(np.histogram(ranks, bins=bins)[0])
    
    #print(np.array( resulting_bin_arrays[species_code][species_code]))
    print(ranks)
    print("done with {} queries".format(ctr))
print('done with all queries')


# ### Searching across different species

# In[ ]:


# The searches across different species.
rank_for_not_found = 100
bins =[0,11,21,31,41,51,rank_for_not_found]
bin_names = [10,20,30,40,50,rank_for_not_found]
assert len(bin_names) == len(bins)-1

ctr = 0
for gene in genes.itertuples():
    
    ctr = ctr+1
    limit = 50
    from_species = gene[1]
    from_species_code = gene[2]
    identifier = gene[3]
    gene_id = gene[5]
    text = texts[gene_id]
    
    
    # Do the switch to make the species for the query and IDs of interest the opposite one.
    to_species = {"Arabidopsis":"Maize","Maize":"Arabidopsis"}[from_species]
    to_species_code = {"ath":"zma","zma":"ath"}[from_species_code]
    
    # Because these are being passed as strings to the command line, quotes need to be removed now,
    # isntead of waiting for them to be removed as a preprocessing step of the search strings in the streamlit script.
    text = text.replace("'","")
    text = text.replace('"','')
    
    
    path = "/Users/irbraun/phenologs-with-oats/quoats/outputs_across_anthocyanin/output_{}.tsv".format(ctr)
    os.chdir('/Users/irbraun/quoats')
    os.system("python main.py -s {} -t identifiers -q '{}:{}' -l {} -o {} -r 0.000 -a TFIDF".format(to_species,from_species,identifier,limit,path))
    time.sleep(4)
    df = pd.read_csv(path, sep='\t')
    df = df[["Rank","Internal ID"]]
    df = df.drop_duplicates()
    id_to_rank = dict(zip(df["Internal ID"].values,df["Rank"].values))
    assert rank_for_not_found > limit
    
    # For within the same species, get rid of the identical gene (always rank 1).
    ids_of_interest = [i for i in genes[genes["species"]==to_species]["id"].values if i != gene_id]
    ranks = [id_to_rank.get(i, rank_for_not_found) for i in ids_of_interest]
    resulting_bin_arrays[from_species_code][to_species_code].append(np.histogram(ranks, bins=bins)[0])
    
    #print(np.array( resulting_bin_arrays[species_code][species_code]))
    print(ranks)
    print("done with {} queries".format(ctr))
print('done with all queries')


# In[ ]:


# Create the output dataframe with the means and standard deviation for each bin and direction.
output_rows = []
for (s1,s2) in product(["ath","zma"],["ath","zma"]):
    means = np.mean(np.array(resulting_bin_arrays[s1][s2]),axis=0)
    std_devs = np.std(np.array(resulting_bin_arrays[s1][s2]),axis=0)
    for i in range(len(bin_names)):
        output_rows.append([s1, s2, bin_names[i], means[i], std_devs[i]])    
names = ["from","to","bin","mean","sd"]    
output_df = pd.DataFrame(output_rows,columns=names)
output_df


# In[ ]:


os.chdir('/Users/irbraun/phenologs-with-oats/quoats')
output_path = "plots/anthocyanin_plot_data.csv"
output_df.to_csv(output_path, index=False)


# In[ ]:


print(stophere)


# ### Autophagy genes

# In[ ]:


mapping = dataset.get_species_to_name_to_ids_dictionary(include_synonyms=False, lowercase=True)
genes = pd.read_csv("plots/autophagy_core_genes.csv")
genes["id"] = genes.apply(lambda x: mapping[x["species_code"]].get(x["identifier"].strip().lower(),-1), axis=1)
genes[genes["id"]!=-1]["id"] = genes[genes["id"]!=-1]["id"].map(lambda x: x[0])
genes["in_current_dataset"] = genes["id"].map(lambda x: x!=-1)
genes

# Looking just at the ones that are in the current dataset.
genes = genes[genes["in_current_dataset"]]
genes["id"] = genes["id"].map(lambda x: x[0])
genes


# 12 out of the 16 core autophagy genes from the list are present in the dataset, see the core genes file.

# In[ ]:


# Grabbing the texts dictionary from the dataset that we can use to grab the descriptions to query.
texts = dataset.get_description_dictionary()
texts[4688]


# In[ ]:


# Prepare dictionaries to hold the resulting arrays.
resulting_bin_arrays = defaultdict(dict)
resulting_bin_arrays["ath"]["ath"] = []
resulting_bin_arrays


# In[ ]:


# The searches within the same species.
rank_for_not_found = 100
bins =[0,11,21,31,41,51,rank_for_not_found]
bin_names = [10,20,30,40,50,rank_for_not_found]
assert len(bin_names) == len(bins)-1

ctr = 0
for gene in genes.itertuples():
    
    ctr = ctr+1
    limit = 50
    species = gene[1]
    species_code = gene[2]
    identifier = gene[3]
    gene_id = gene[5]
    text = texts[gene_id]
    
    # Because these are being passed as strings to the command line, quotes need to be removed now,
    # isntead of waiting for them to be removed as a preprocessing step of the search strings in the streamlit script.
    text = text.replace("'","")
    text = text.replace('"','')
    
    
    
    path = "/Users/irbraun/phenologs-with-oats/quoats/outputs_within_autophagy/output_{}.tsv".format(ctr)
    os.chdir('/Users/irbraun/quoats')
    os.system("python main.py -s {} -t identifiers -q '{}:{}' -l {} -o {} -r 0.000 -a tfidf".format(to_species,from_species,identifier,limit,path))
    time.sleep(4)
    df = pd.read_csv(path, sep='\t')
    df = df[["Rank","Internal ID"]]
    df = df.drop_duplicates()
    id_to_rank = dict(zip(df["Internal ID"].values,df["Rank"].values))
    assert rank_for_not_found > limit
    
    
    # For within the same species, get rid of the identical gene (always rank 1).
    ids_of_interest = [i for i in genes[genes["species"]==species]["id"].values if i != gene_id]
    ranks = [id_to_rank.get(i, rank_for_not_found) for i in ids_of_interest]
    resulting_bin_arrays[species_code][species_code].append(np.histogram(ranks, bins=bins)[0])
    
    #print(np.array( resulting_bin_arrays[species_code][species_code]))
    print(ranks)
    print("done with {} queries".format(ctr))
print('done with all queries')


# In[ ]:


# Create the output dataframe with the means and standard deviation for each bin and direction.
output_rows = []
s1 = "ath"
s2 = "ath"
means = np.mean(np.array(resulting_bin_arrays[s1][s2]),axis=0)
std_devs = np.std(np.array(resulting_bin_arrays[s1][s2]),axis=0)
for i in range(len(bin_names)):
    output_rows.append([s1, s2, bin_names[i], means[i], std_devs[i]])    
names = ["from","to","bin","mean","sd"]    
output_df = pd.DataFrame(output_rows,columns=names)
output_df


# In[ ]:


os.chdir('/Users/irbraun/phenologs-with-oats/quoats')
output_path = "plots/autophagy_plot_data.csv"
output_df.to_csv(output_path, index=False)

