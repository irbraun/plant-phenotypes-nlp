#!/usr/bin/env python
# coding: utf-8

# # Table of Contents
# 
# - [Introduction](#introduction)
# 
# 
# - [Links of Interest](#links)
# 
# 
# - [Part 1. Loading and Filtering Data](#part_1)
#     - [Reading in arguments](#args)
#     - [Setting input and output paths](#paths)
#     - [Reading in genes, annotations, and phenotype descriptions](#read_text_data)
#     - [Relating genes in this dataset to other biological datasets](#relating)
#     - [KEGG](#kegg)
#     - [PlantCyc](#plantcyc)
#     - [Lloyd and Meinke (2012) phenotype subsets](#subsets)
#     - [Lloyd and Meinke (2012) phenotype classes](#classes)
#     - [Oellrich, Walls et al., (2015) EQ statements](#eqs)
#     - [Protein associations from STRING](#string)
#     - [Ortholog relationships from PANTHER](#panther)
#     - [Filtering the dataset to include relevant genes](#filtering)
#     
#         
# - [Part 2. NLP Models](#part_2)
#     - [Word2Vec and Doc2Vec](#word2vec_doc2vec)
#     - [BERT and BioBERT](#bert_biobert)
#     - [Loading models](#load_models)
# 
# 
# - [Part 3. NLP Choices](#part_3)
#     - [Preprocessing descriptions](#preprocessing)
#     - [POS Tagging](#pos_tagging)
#     - [Reducing vocabulary size](#vocab)
#     - [Annotating with biological ontologies](#annotation)
#     - [Splitting into phene descriptions](#phenes)
#     
#     
# - [Part 4. Generating Vectors and Distance Matrices](#part_4)
#     - [Defining methods to use](#methods)
#     - [Running all methods](#running)
#     - [Merging distances into an edgelist](#merging)
#     
#     
# - [Part 5. Biological Questions](#part_5)
#     - [Using pathways as the objective](#pathway_objective)
#     - [Using phenotype subsets as the objective](#subset_objective)
#     - [Using protein associations as the objective](#association_objective)
#     - [Using orthology as the objective](#ortholog_objective)
#     - [Adding EQ similarity values](#eq_sim)
#     - [Noting whether gene pairs have curated data](#curated)
#     - [Noting whether gene pairs refer to the same species](#species)
#     - [Determining the number of genes and pairs involved in each question](#n_values)
#     - [Determining how similar the biological questions are to one another](#objective_similarities)
#     
# 
# - [Part 6. Results](#part_6)
#     - [Distributions of distance values](#ks)
#     - [Within-group distance values](#within)
#     - [Predictions and AUC for shared pathways or interactions](#auc)
#     - [Tests for querying to recover related genes](#y)
#     - [Producing output summary table](#output)
# 
# 
# - [Part 6. Clustering Analysis](#part_7)
#     - [Topic modeling](#topic_modeling)
#     - [Agglomerative clustering](#clustering)
#     - [Phenologs for OMIM disease phenotypes](#phenologs)
#     
# 

# <a id="introduction"></a>
# ### Introduction: Text Mining Analysis of Phenotype Descriptions in Plants
# The purpose of this notebook is to evaluate what can be learned from a natural language processing approach to analyzing free-text descriptions of phenotype descriptions of plants. The approach is to generate pairwise distances matrices between a set of plant phenotype descriptions across different species, sourced from academic papers and online model organism databases. These pairwise distance matrices can be constructed using any vectorization method that can be applied to natural language. In this notebook, we specifically evaluate the use of n-gram and bag-of-words techniques, word and document embedding using Word2Vec and Doc2Vec, context-dependent word-embeddings using BERT and BioBERT, and ontology term annotations with automated annotation tools such as NOBLE Coder.
# 
# Loading, manipulation, and filtering of the dataset of phenotype descriptions associated with genes across different plant species is largely handled through a Python package created for this purpose called OATS (Ontology Annotation and Text Similarity) which is available [here](https://github.com/irbraun/oats). Preprocessing of the descriptions, mapping the dataset to additional resources such as protein-protein interaction databases and biochemical pathway databases are handled in this notebook using that package as well. In the evaluation of each of these natural language processing approaches to analyzing this dataset of descriptions, we compare performance against a dataset generated through manual annotation of a similar dataset in Oellrich Walls et al. (2015) and against manual annotations with experimentally determined terms from the Gene Ontology (PO) and the Plant Ontology (PO).
# 
# <a id="links"></a>
# ### Relevant links of interest:
# - Paper describing comparison of NLP and ontology annotation approaches to curation: [Braun, Lawrence-Dill (2019)](https://doi.org/10.3389/fpls.2019.01629)
# - Paper describing results of manual phenotype description curation: [Oellrich, Walls et al. (2015](https://plantmethods.biomedcentral.com/articles/10.1186/s13007-015-0053-y)
# - Plant databases with phenotype description text data available: [TAIR](https://www.arabidopsis.org/), [SGN](https://solgenomics.net/), [MaizeGDB](https://www.maizegdb.org/)
# - Python package for working with phenotype descriptions: [OATS](https://github.com/irbraun/oats)
# - Python package used for general NLP functions: [NLTK](https://www.nltk.org/), [Gensim](https://radimrehurek.com/gensim/auto_examples/index.html)
# - Python package used for working with biological ontologies: [Pronto](https://pronto.readthedocs.io/en/latest/)
# - Python package for loading pretrained BERT models: [PyTorch Pretrained BERT](https://pypi.org/project/pytorch-pretrained-bert/)
# - For BERT Models pretrained on PubMed and PMC: [BioBERT Paper](https://arxiv.org/abs/1901.08746), [BioBERT Models](https://github.com/naver/biobert-pretrained)

# In[1]:


import datetime
import nltk
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np
import time
import math
import sys
import gensim
import os
import warnings
import torch
import itertools
import argparse
import shlex
import random
import multiprocessing as mp
from collections import Counter, defaultdict
from inspect import signature
from scipy.stats import ks_2samp, hypergeom, pearsonr, spearmanr
from sklearn.metrics import precision_recall_curve, f1_score, auc
from sklearn.model_selection import train_test_split, KFold
from scipy import spatial, stats
from statsmodels.sandbox.stats.multicomp import multipletests
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from gensim.parsing.preprocessing import strip_non_alphanum, stem_text, preprocess_string, remove_stopwords
from gensim.utils import simple_preprocess
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.cluster import AgglomerativeClustering

from nltk.corpus import brown, stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('brown', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

sys.path.append("../../oats")
from oats.utils.utils import save_to_pickle, load_from_pickle, merge_list_dicts, flatten, to_hms
from oats.utils.utils import function_wrapper_with_duration
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

from _utils import Method
from _utils import IndexedGraph

mpl.rcParams["figure.dpi"] = 400
warnings.simplefilter('ignore')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# <a id="part_1"></a>
# # Part 1. Loading and Filtering Data
# This section defines some constants which are used for creating a uniquely named directory to contain all the outputs from running this instance of this notebook. The naming scheme is based on the time that the notebook is run. All the input and output file paths for loading datasets or models are also contained within this cell, so that if anything is moved the directories and file names should only have to be changed at this point and nowhere else further into the notebook. If additional files are added to the notebook cells they should be put here as well.

# <a id="args"></a>
# ### Reading in arguments

# In[91]:


NOTEBOOK = False

parser = argparse.ArgumentParser()
parser.add_argument("--learning", dest="learning", required=False, action='store_true', help="use the approaches that involve neural networks")
parser.add_argument("--noblecoder", dest="noblecoder", required=False, action='store_true', help="use the approaches that involve computational annotation")
parser.add_argument("--lda", dest="lda", required=False, action='store_true', help="use the approaches that involve topic modeling")
parser.add_argument("--nmf", dest="nmf", required=False, action='store_true', help="use the approaches that involve topic modeling")
parser.add_argument("--vanilla", dest="vanilla", required=False, action='store_true', help="use the n-grams (bag-of-words) approach")
parser.add_argument("--vocab", dest="vocab", required=False, action='store_true', help="using the n-grams approach but with modified vocabularies")
parser.add_argument("--annotations", dest="annotations", required=False, action='store_true', help="use the curated annotations")

if NOTEBOOK:
    arg_string = "--learning --noblecoder --lda --nmf --vanilla --vocab --annotations"
    args = parser.parse_args(shlex.split(arg_string))
else:
    args = parser.parse_args()


# <a id="paths"></a>
# ### Defining the input file paths and creating output directory

# In[3]:


# Create and name an output directory according to when the notebooks or script was run.
OUTPUT_DIR = os.path.join("../outputs","{}_r{}".format(datetime.datetime.now().strftime('%m_%d_%Y_h%Hm%Ms%S'),random.randrange(1000,9999)))
os.mkdir(OUTPUT_DIR)


# In[4]:


dataset_filename = "../data/pickles/gene_phenotype_dataset_all_text_and_annotations.pickle"          # The full dataset pickle.
kegg_pathways_filename = "../data/pickles/groupings_from_kegg_pathways.pickle"                       # The pathway groupings from KEGG.
pmn_pathways_filename = "../data/pickles/groupings_from_pmn_pathways.pickle"                         # The pahway groupings from Plant Metabolic Network.
lloyd_subsets_filename = "../data/pickles/groupings_from_lloyd_subsets.pickle"                       # The functional subsets defined by Lloyd and Meinke (2012).
lloyd_classes_filename = "../data/pickles/groupings_from_lloyd_classes.pickle"                       # The functional classes defined by Lloyd and Meinke (2012).
background_corpus_filename = "../data/corpus_related_files/untagged_text_corpora/background.txt"     # Text file with background content.
phenotypes_corpus_filename = "../data/corpus_related_files/untagged_text_corpora/phenotypes_all.txt" # Text file with specific content.
doc2vec_pubmed_filename = "../gensim/pubmed_dbow/doc2vec_2.bin"                                      # File holding saved Doc2Vec model trained on PubMed.
doc2vec_wikipedia_filename = "../gensim/enwiki_dbow/doc2vec.bin"                                     # File holding saved Doc2Vec model trained on Wikipedia.
word2vec_model_filename = "../gensim/wiki_sg/word2vec.bin"                                           # File holding saved Word2Vec model trained on Wikipedia.
go_filename = "../ontologies/go.obo"                                                                 # Gene Ontology file in OBO format.
po_filename = "../ontologies/po.obo"                                                                 # Plant Ontology file in OBO format.
pato_filename = "../ontologies/pato.obo"                                                             # Phenotype and Trait Ontology file in OBO format.
noblecoder_jarfile_path = "../lib/NobleCoder-1.0.jar"                                                # Jar for NOBLE Coder annotation tool.
biobert_pmc_path = "../gensim/biobert_v1.0_pmc/pytorch_model"                                        # Path for PyTorch BioBERT model.
biobert_pubmed_path = "../gensim/biobert_v1.0_pubmed/pytorch_model"                                  # Path for PyTorch BioBERT model.
biobert_pubmed_pmc_path = "../gensim/biobert_v1.0_pubmed_pmc/pytorch_model"                          # Path for PyTorch BioBERT model.
panther_to_omim_filename = "../data/orthology_related_files/ath_to_hsa/pantherdb_omim_df.csv"        # File with mappings to human orthologs and disease phenotypes.
pppn_edgelist_path = "../data/supplemental_files_oellrich_walls/13007_2015_53_MOESM9_ESM.txt"
ortholog_file_path = "../data/orthology_related_files/pantherdb/PlantGenomeOrthologs_IRB_Modified.txt"
paired_phenotypes_path = "../data/corpus_related_files/phenotype_pairs/scored.csv"
lloyd_function_hierarchy_path = "../data/group_related_files/lloyd/lloyd_function_hierarchy_irb_cleaned.csv"


# <a id="read_text_data"></a>
# ### Reading in the dataset of genes and their associated phenotype descriptions and annotations

# In[5]:


dataset = load_from_pickle(dataset_filename)
dataset.filter_has_description()
dataset.describe()


# <a id="relating"></a>
# ### Relating the dataset of genes to the dataset of groupings or categories
# This section generates tables that indicate how the genes present in the dataset were mapped to the defined pathways or groups. This includes a summary table that indicates how many genes by species were succcessfully mapped to atleast one pathway or group, as well as a more detailed table describing how many genes from each species were mapped to each particular pathway or group. Additionally, a pairwise group similarity matrix is also generated, where the similarity is given as the Jaccard similarity between two groups based on whether genes are shared by those groups or not. The function defined in this section returns a groupings object that can be re-used, as well as the IDs of the genes in the full dataset that were found to be relevant to those particular groupings.

# In[6]:


def read_in_groupings_object_and_write_summary_tables(dataset, groupings_filename, name):

    # Load the groupings object.
    groups = load_from_pickle(groupings_filename)
    id_to_group_ids = groups.get_id_to_group_ids_dict(dataset.get_gene_dictionary())
    group_id_to_ids = groups.get_group_id_to_ids_dict(dataset.get_gene_dictionary())
    group_mapped_ids = [k for (k,v) in id_to_group_ids.items() if len(v)>0]
    groups.to_csv(os.path.join(OUTPUT_DIR,"part_1_{}_groupings.csv".format(name)))

    # Generate a table describing how many of the genes input from each species map to atleast one group.
    summary = defaultdict(dict)
    species_dict = dataset.get_species_dictionary()
    for species in dataset.get_species():
        summary[species]["input"] = len([x for x in dataset.get_ids() if species_dict[x]==species])
        summary[species]["mapped"] = len([x for x in group_mapped_ids if species_dict[x]==species])
    table = pd.DataFrame(summary).transpose()
    table.loc["total"]= table.sum()
    table["fraction"] = table.apply(lambda row: "{:0.4f}".format(row["mapped"]/row["input"]), axis=1)
    table = table.reset_index(inplace=False)
    table = table.rename({"index":"species"}, axis="columns")
    table.to_csv(os.path.join(OUTPUT_DIR,"part_1_{}_mappings_summary.csv".format(name)), index=False)

    
    # Generate a table describing how many genes from each species map to which particular group.
    summary = defaultdict(dict)
    for group_id,ids in group_id_to_ids.items():
        summary[group_id].update({species:len([x for x in ids if species_dict[x]==species]) for species in dataset.get_species()})
        summary[group_id]["total"] = len([x for x in ids])
    table = pd.DataFrame(summary).transpose()
    table = table.sort_values(by="total", ascending=False)
    table = table.reset_index(inplace=False)
    table = table.rename({"index":"pathway_id"}, axis="columns")
    table["pathway_name"] = table["pathway_id"].map(groups.get_long_name)
    table.loc["total"] = table.sum()
    table.loc["total","pathway_id"] = "total"
    table.loc["total","pathway_name"] = "total"
    table = table[table.columns.tolist()[-1:] + table.columns.tolist()[:-1]]
    table.to_csv(os.path.join(OUTPUT_DIR,"part_1_{}_mappings_by_group.csv".format(name)), index=False)
    
    
    # What are the similarites between the groups for the genes present in this dataset?
    group_sims = defaultdict(dict)
    for group_id_1,ids_1 in group_id_to_ids.items():
        for group_id_2,ids_2 in group_id_to_ids.items():
            jaccard_sim = len(set(ids_1).intersection(set(ids_2)))/len(set(ids_1).union(set(ids_2)))
            group_sims[group_id_1][group_id_2] = jaccard_sim
    table = pd.DataFrame(group_sims)
    
    
    # Changing the order of the Lloyd, Meinke phenotype subsets to match other figures for consistency, special case.
    if name == "subsets":
        filename = "../data/group_related_files/lloyd/lloyd_function_hierarchy_irb_cleaned.csv"
        lmtm_df = pd.read_csv(filename)    
        subsets_in_order = [col for col in lmtm_df["Subset Symbol"].values if col in table.columns]
        table = table[subsets_in_order]
        table = table.reindex(subsets_in_order)
        
        
    # Formatting the column names for this table correctly and outputting to a file.
    table = table.reset_index(drop=False).rename({"index":"group"},axis=1).reset_index(drop=False).rename({"index":"order"},axis=1)
    table.to_csv(os.path.join(OUTPUT_DIR,"part_1_{}_similarity_matrix.csv".format(name)), index=False)
    

    # Returning the groupings object and the list of IDs for genes that were mapped to one or more groups.
    return(groups, group_mapped_ids)


# <a id="kegg"></a>
# ### Reading in and relating the pathways from KEGG

# In[7]:


# Readin in the dataset of groupings for pathways in KEGG.
kegg_groups, kegg_mapped_ids = read_in_groupings_object_and_write_summary_tables(dataset, kegg_pathways_filename, "kegg")
kegg_groups.to_pandas().head(20)


# <a id="plantcyc"></a>
# ### Reading in and relating the pathways from PlantCyc

# In[8]:


# Reading in the dataset of groupings for pathways in PlantCyc.
pmn_groups, pmn_mapped_ids = read_in_groupings_object_and_write_summary_tables(dataset, pmn_pathways_filename, "pmn")
pmn_groups.to_pandas().head(20)


# <a id="subsets"></a>
# ###  Reading in and relating the phenotype subsets from Lloyd and Meinke (2012)

# In[9]:


# Reading in the datasets of phenotype subset classifications from the Lloyd, Meinke 2012 paper.
phe_subsets_groups, subsets_mapped_ids = read_in_groupings_object_and_write_summary_tables(dataset, lloyd_subsets_filename, "subsets")
phe_subsets_groups.to_pandas().head(20)


# <a id="classes"></a>
# ### Reading in and relating the phenotype subsets from Lloyd and Meinke (2012)

# In[10]:


# Reading in the datasets of phenotype class classifications from the Lloyd, Meinke 2012 paper.
phe_classes_groups, classes_mapped_ids = read_in_groupings_object_and_write_summary_tables(dataset, lloyd_classes_filename, "classes")
phe_classes_groups.to_pandas().head(20)


# ### Relating pairs of genes to information about network edges from other sources
# This is done to only include genes (and the corresponding phenotype descriptions and annotations) which are useful for the current analysis. In this case we want to only retain genes that are mentioned atleast one time in the STRING database for a given species. If a gene is not mentioned at all in STRING, there is no information available for whether or not it interacts with any other proteins in the dataset so choose to not include it in the analysis. Only genes that have atleast one true positive are included because these are the only ones for which the missing information (negatives) is meaningful. This should be run instead of the subsequent cell, or the other way around, based on whether or not protein-protein interactions is the prediction goal for the current analysis.

# <a id="eqs"></a>
# ### EQ-based similarities from Oellrich, Walls et al., (2015)

# In[11]:


ow_edgelist = AnyInteractions(dataset.get_name_to_id_dictionary(), pppn_edgelist_path)
ow_edgelist.df.head(10)


# <a id="string"></a>
# ### Protein associations from STRING

# In[12]:


naming_file = "../data/group_related_files/string/all_organisms.name_2_string.tsv"
interaction_files = [
    "../data/group_related_files/string/3702.protein.links.detailed.v11.0.txt", # Arabidopsis
    "../data/group_related_files/string/4577.protein.links.detailed.v11.0.txt", # Maize
    "../data/group_related_files/string/4530.protein.links.detailed.v11.0.txt", # Tomato 
    "../data/group_related_files/string/4081.protein.links.detailed.v11.0.txt", # Medicago
    "../data/group_related_files/string/3880.protein.links.detailed.v11.0.txt", # Rice 
    "../data/group_related_files/string/3847.protein.links.detailed.v11.0.txt", # Soybean
]
genes = dataset.get_gene_dictionary()
string_edgelist = ProteinInteractions(genes, naming_file, *interaction_files)
string_edgelist.df.head(10)


# <a id="panther"></a>
# ### Orthologous genes from PANTHER

# In[13]:


panther_edgelist = AnyInteractions(dataset.get_name_to_id_dictionary(), ortholog_file_path)
panther_edgelist.df.head(10)


# <a id="filtering"></a>
# ### Subsetting the dataset to include only genes with relevance to any of the biological questions
# This is done to only include genes (and the corresponding phenotype descriptions and annotations) which are useful for the current analysis. In this case we want to only retain genes that are mapped to atleast one pathway in whatever the source of pathway membership we are using is (KEGG, Plant Metabolic Network, etc). This is because for these genes, it will be impossible to correctly predict their pathway membership, and we have no evidence that they belong or do not belong in certain pathways so they can not be identified as being true or false negatives in any case. This should not actually be necessary if the dataset used to start the notebook analysis has already be subset for just the genes that either have pathway information of phenotype classification information, this should just be used to double check that the numbers make sense.

# In[14]:


# Get the list of all the IDs in this dataset that have any relevant mapping at all to the biological questions.
ids_with_any_mapping = list(set(flatten([
    kegg_mapped_ids,
    pmn_mapped_ids,
    subsets_mapped_ids,
    classes_mapped_ids,
    string_edgelist.ids,
    panther_edgelist.ids
])))


# In[15]:


# Get the list of all the IDs in this dataset that have all of types of curated values we want to look at. 
annots = dataset.get_annotations_dictionary()
go_mapped_ids = [i for i in dataset.get_ids() if "GO" in annots[i]]
po_mapped_ids = [i for i in dataset.get_ids() if "PO" in annots[i]]
ids_with_all_annotations = list(set(flatten([
    go_mapped_ids,
    po_mapped_ids,
    ow_edgelist.ids
])))


# In[16]:


dataset.filter_with_ids(ids_with_any_mapping)
#dataset.filter_random_k(300)
dataset.describe()


# <a id="part_2"></a>
# # Part 2. NLP Models
# 
# 
# <a id="word2vec_doc2vec"></a>
# ### Word2Vec and Doc2Vec
# Word2Vec is a word embedding technique using a neural network trained on a so-called *false task*, namely either predicting a missing word from within a sequence of context words drawn from a sentence or phrase, or predicting which contexts words surround some given input word drawn from a sentence or phrase. Each of these tasks are supervised (the correct answer is fixed and known), but can be generated from unlabelled text data such as a collection of books or wikipedia articles, meaning that even though the task itself is supervised the training data can be generated automatically, enabling the creation of enormous training sets. The internal representation for particular words learned during the training process contain semantically informative features related to that given word, and can therefore be used as embeddings used downstream for tasks such as finding similarity between words or as input into additional models. Doc2Vec is an extension of this technique that determines vector embeddings for entire documents (strings containing multiple words, could be sentences, paragraphs, or documents).
# 
# 
# <a id="bert_biobert"></a>
# ### BERT and BioBERT
# BERT ('Bidirectional Encoder Representations from Transformers') is another neueral network-based model trained on two different false tasks, namely predicting the subsequent sentence given some input sentence, or predicting the identity of a set of words masked from an input sentence. Like Word2Vec, this architecture can be used to generate vector embeddings for a particular input word by extracting values from a subset of the encoder layers that correspond to that input word. Practically, a major difference is that because the input word is input in the context of its surrounding sentence, the embedding reflects the meaning of a particular word in a particular context (such as the difference in the meaning of *root* in the phrases *plant root* and *root of the problem*. BioBERT refers to a set of BERT models which have been finetuned on the PubMed and PMC corpora. See the list of relevant links for the publications and pages associated with these models.
# 
# <a id="load_models"></a>
# ### Loading trained and saved models
# Versions of the architectures discussed above which have been saved as trained models are loaded here. Some of these models are loaded as pretrained models from the work of other groups, and some were trained on data specific to this notebook and loaded here.

# In[17]:


# Files and models related to the machine learning text embedding methods used here.
doc2vec_wiki_model = gensim.models.Doc2Vec.load(doc2vec_wikipedia_filename)
doc2vec_pubmed_model = gensim.models.Doc2Vec.load(doc2vec_pubmed_filename)
word2vec_model = gensim.models.Word2Vec.load(word2vec_model_filename)
bert_tokenizer_base = BertTokenizer.from_pretrained('bert-base-uncased')
bert_tokenizer_pmc = BertTokenizer.from_pretrained(biobert_pmc_path)
bert_tokenizer_pubmed = BertTokenizer.from_pretrained(biobert_pubmed_path)
bert_tokenizer_pubmed_pmc = BertTokenizer.from_pretrained(biobert_pubmed_pmc_path)
bert_model_base = BertModel.from_pretrained('bert-base-uncased')
bert_model_pmc = BertModel.from_pretrained(biobert_pmc_path)
bert_model_pubmed = BertModel.from_pretrained(biobert_pubmed_path)
bert_model_pubmed_pmc = BertModel.from_pretrained(biobert_pubmed_pmc_path)


# ### Reading in the descriptions from hand-picked dataset of phenotype pairs
# See the other notebook for the creation of this dataset. This is included in this notebook instead of a separated notebook because we want the treatment of the individual phenotype text instances to be the same as is done for the descriptions from the real dataset of plant phenotypes. The list of computational approaches being evaluated for this task is the same in both cases so all of the cells between the point where the descriptions are read in and when the distance matrices are found using all those methods are the same for this task as any of the biological questions that this notebook is focused on.

# In[18]:


# Read in the table of similarity scored phenotype pairs that was prepared from random selection.
num_pairs = 50
mupdata = pd.read_csv(paired_phenotypes_path)
assert num_pairs == mupdata.shape[0]
paired_descriptions = mupdata["Phenotype 1"].values.tolist()
paired_descriptions.extend(mupdata["Phenotype 2"].values.tolist())
first_paired_id = 0
paired_descriptions = {i:description for i,description in enumerate(paired_descriptions, first_paired_id)}
pair_to_score = {(i,i+num_pairs):s for i,s in enumerate(mupdata["Score"].values, first_paired_id)}
paired_ids = list(paired_descriptions.keys())

# Set the descriptions to be used to be these from the paired phenotypes dataset.
# This will only matter if running this in the context of the notebook and skipping the other steps.
# Otherwise the descriptions dictionary will be reset to be the descriptions from the full dataset.
descriptions = paired_descriptions


# <a id="part_3"></a>
# # Part 3. NLP Choices

# In[19]:


# Obtain a mapping between IDs and the raw text descriptions associated with that ID from the dataset.
descriptions = dataset.get_description_dictionary()


# ### Converting phenotype descriptions to phene descriptions and combining for batch preprocessing steps

# In[20]:


# Create a dictionary of phene descriptions and a dictionary to convert back to the phenotype/gene IDs.
phenotype_ids = list(descriptions.keys())
phenes = {}
phene_id_to_id = {}
phene_id = max(phenotype_ids)+1
for i,phene_list in {i:sent_tokenize(d) for i,d in descriptions.items()}.items():
    for phene in phene_list:
        phenes[phene_id] = phene
        phene_id_to_id[phene_id] = i
        phene_id = phene_id+1
phene_ids = list(phenes.keys())


# Update the dictionary of descriptions to also include the individualy parsed phenes.
descriptions.update(phenes)
assert len(descriptions) == len(phenotype_ids)+len(phene_ids)


# <a id="preprocessing"></a>
# ### Preprocessing text descriptions
# The preprocessing methods applied to the phenotype descriptions are a choice which impacts the subsequent vectorization and similarity methods which construct the pairwise distance matrix from each of these descriptions. The preprocessing methods that make sense are also highly dependent on the vectorization method or embedding method that is to be applied. For example, stemming (which is part of the full proprocessing done below using the Gensim preprocessing function) is useful for the n-grams and bag-of-words methods but not for the document embeddings methods which need each token to be in the vocabulary that was constructed and used when the model was trained. For this reason, embedding methods with pretrained models where the vocabulary is fixed should have a lighter degree of preprocessing not involving stemming or lemmatization but should involve things like removal of non-alphanumerics and normalizing case. 

# In[21]:


# Applying canned prepreprocessing approaches to the descriptions.
processed = defaultdict(dict)
processed["simple"] = {i:" ".join(simple_preprocess(d)) for i,d in descriptions.items()}
processed["simple_no_stops"] = {i:remove_stopwords(" ".join(simple_preprocess(d))) for i,d in descriptions.items()}
processed["full"] = {i:" ".join(preprocess_string(d)) for i,d in descriptions.items()}

# Set of stopwords, used later for checking it tokens in a list are stopwords or not.
stop_words = set(stopwords.words('english')) 


# <a id="pos_tagging"></a>
# ### POS tagging the phenotype descriptions for nouns and adjectives
# Note that preprocessing of the descriptions should be done after part-of-speech tagging, because tokens that are removed during preprocessing before n-gram analysis contain information that the parser needs to accurately call parts-of-speech. This step should be done on the raw descriptions and then the resulting bags of words can be subset using additional preprocesssing steps before input in one of the vectorization methods.

# In[22]:


get_pos_tokens = lambda text,pos: " ".join([t[0] for t in nltk.pos_tag(word_tokenize(text)) if t[1].lower()==pos.lower()])
processed["nouns"] =  {i:get_pos_tokens(d,"NN") for i,d in descriptions.items()}
processed["nouns_full"] = {i:" ".join(preprocess_string(d)) for i,d in processed["nouns"].items()}
processed["nouns_simple"] = {i:" ".join(simple_preprocess(d)) for i,d in processed["nouns"].items()}
processed["adjectives"] =  {i:get_pos_tokens(d,"JJ") for i,d in descriptions.items()}
processed["adjectives_full"] = {i:" ".join(preprocess_string(d)) for i,d in processed["adjectives"].items()}
processed["adjectives_simple"] = {i:" ".join(simple_preprocess(d)) for i,d in processed["adjectives"].items()}
processed["nouns_adjectives"] = {i:"{} {}".format(processed["nouns"][i],processed["adjectives"][i]) for i in descriptions.keys()}
processed["nouns_adjectives_full"] = {i:"{} {}".format(processed["nouns_full"][i],processed["adjectives_full"][i]) for i in descriptions.keys()}
processed["nouns_adjectives_simple"] = {i:"{} {}".format(processed["nouns_simple"][i],processed["adjectives_simple"][i]) for i in descriptions.keys()}


# ### Reducing vocabulary size based on identifying important words
# These approcahes for reducing the vocabulary size of the dataset work by identifying which words in the descriptions are likely to be the most important for identifying differences between the phenotypes and meaning of the descriptions. One approach is to determine which words occur at a higher rate in text of interest such as articles about plant phenotypes as compared to their rates in more general texts such as a corpus of news articles. These approaches do not create modified versions of the descriptions but rather provide vocabulary objects that can be passed to the sklearn vectorizer or constructors.

# In[23]:


# Create ontology objects for all the biological ontologies being used.
pato = Ontology(pato_filename)
po = Ontology(po_filename)
go = Ontology(go_filename)


# In[24]:


# Getting sets of tokens that are part of bio ontology term labels or synonyms.
bio_ontology_tokens = list(set(po.tokens()).union(set(go.tokens())))
bio_ontology_tokens = [t for t in bio_ontology_tokens if t not in stop_words]
bio_ontology_tokens_simple = flatten([simple_preprocess(t) for t in bio_ontology_tokens])
bio_ontology_tokens_full = flatten([preprocess_string(t) for t in bio_ontology_tokens])
with open(os.path.join(OUTPUT_DIR,"part_3_bio_ontology_vocab_size_{}.txt".format(len(bio_ontology_tokens))),"w") as f:
    f.write(" ".join(bio_ontology_tokens))


# In[25]:


# Getting sets of tokens that are overprepresented in plant phenotype papers as compared to some background corpus.
maximum_number_of_tokens = 10000
background_corpus = open(background_corpus_filename,"r").read()
phenotypes_corpus = open(phenotypes_corpus_filename,"r").read()
ppp_overrepresented_tokens = get_overrepresented_tokens(phenotypes_corpus, background_corpus, max_features=maximum_number_of_tokens)
ppp_overrepresented_tokens = [t for t in ppp_overrepresented_tokens if t not in stop_words]
ppp_overrepresented_tokens_simple = flatten([simple_preprocess(t) for t in ppp_overrepresented_tokens])
ppp_overrepresented_tokens_full = flatten([preprocess_string(t) for t in ppp_overrepresented_tokens])
with open(os.path.join(OUTPUT_DIR,"part_3_plant_phenotype_vocab_size_{}.txt".format(len(ppp_overrepresented_tokens))), "w") as f:
    f.write(" ".join(ppp_overrepresented_tokens))


# In[26]:


# Generating processed description entries by subsetting tokens to only include ones from these vocabularies.
ppp_overrepresented_tokens_full_set = set(ppp_overrepresented_tokens_full)
bio_ontology_tokens_full_set = set(bio_ontology_tokens_full)
processed["plant_overrepresented_tokens"] = {i:" ".join([token for token in word_tokenize(text) if token in ppp_overrepresented_tokens_full_set]) for i,text in processed["full"].items()}
processed["bio_ontology_tokens"] = {i:" ".join([token for token in word_tokenize(text) if token in bio_ontology_tokens_full_set]) for i,text in processed["full"].items()}


# <a id="vocab"></a>
# ### Reducing the vocabulary size using a word distance matrix
# These approaches for reducing the vocabulary size of the dataset work by replacing multiple words that occur throughout the dataset of descriptions with an identical word that is representative of this larger group of words. The total number of unique words across all descriptions is therefore reduced, and when observing n-gram overlaps between vector representations of these descriptions, overlaps will now occur between descriptions that included different but similar words. These methods work by actually generating versions of these descriptions that have the word replacements present. The returned objects for these methods are the revised description dictionary, a dictionary mapping tokens in the full vocabulary to tokens in the reduced vocabulary, and a dictionary mapping tokens in the reduced vocabulary to a list of tokens in the full vocabulary.

# In[27]:


# Generate a pairwise distance matrix object using the oats subpackage, and create an appropriately shaped matrix,
# making sure that the tokens list is in the same order as the indices representing each word in the distance matrix.
# This is currently triviala because the IDs that are used are ordered integers 0 to n, but this might not always be
# the case so it's not directly assumed here.
tokens = list(set([w for w in flatten(d.split() for d in processed["simple"].values())]))
tokens_dict = {i:w for i,w in enumerate(tokens)}
graph = pw.pairwise_square_word2vec(word2vec_model, tokens_dict, "cosine")
distance_matrix = graph.array
tokens = [tokens_dict[graph.index_to_id[index]] for index in np.arange(distance_matrix.shape[0])]

# The other argument that the Linares Pontes algorithm needs is a value for n, see paper or description above.
n = 3
processed["linares_pontes"], reduce_lp, unreduce_lp = reduce_vocab_linares_pontes(processed["simple"], tokens, distance_matrix, n)


# <a id="annotation"></a>
# ### Annotating descriptions with ontology terms
# This section generates dictionaries that map gene IDs from the dataset to lists of strings, where those strings are ontology term IDs. How the term IDs are found for each gene entry with its corresponding phenotype description depends on the cell below. Firstly, the terms are found by using the NOBLE Coder annotation tool through these wrapper functions to identify the terms by looking for instances of the term's label or synonyms in the actual text of the phenotype descriptions. Secondly, the next cell just draws the terms directly from the dataset itself. In this case, these are high-confidence annotations done by curators for a comparison against what can be accomplished through computational analysis of the text.

# In[28]:


# Run the NOBLE Coder annotator over the raw input text descriptions, which handles things like case normalization.
direct_annots_nc_go_precise = annotate_using_noble_coder(descriptions, noblecoder_jarfile_path, "go", precise=1)
direct_annots_nc_go_partial = annotate_using_noble_coder(descriptions, noblecoder_jarfile_path, "go", precise=0)
direct_annots_nc_po_precise = annotate_using_noble_coder(descriptions, noblecoder_jarfile_path, "po", precise=1)
direct_annots_nc_po_partial = annotate_using_noble_coder(descriptions, noblecoder_jarfile_path, "po", precise=0)
direct_annots_nc_pato_precise = annotate_using_noble_coder(descriptions, noblecoder_jarfile_path, "pato", precise=1)
direct_annots_nc_pato_partial = annotate_using_noble_coder(descriptions, noblecoder_jarfile_path, "pato", precise=0)

# Use the ontology hierarchies to add terms that are inherited by the terms that were annotated to the text.
inherited_annots_nc_go_precise = {i:go.inherited(term_id_list) for i,term_id_list in direct_annots_nc_go_precise.items()}
inherited_annots_nc_go_partial = {i:go.inherited(term_id_list) for i,term_id_list in direct_annots_nc_go_partial.items()}
inherited_annots_nc_po_precise = {i:po.inherited(term_id_list) for i,term_id_list in direct_annots_nc_po_precise.items()}
inherited_annots_nc_po_partial = {i:po.inherited(term_id_list) for i,term_id_list in direct_annots_nc_po_partial.items()}
inherited_annots_nc_pato_precise = {i:pato.inherited(term_id_list) for i,term_id_list in direct_annots_nc_pato_precise.items()}
inherited_annots_nc_pato_partial = {i:pato.inherited(term_id_list) for i,term_id_list in direct_annots_nc_pato_partial.items()}

# Merge the ontology term annotations for each descritpion into a single dictionary for the precise and partial levels.
all_precise_annotations = {i:flatten([inherited_annots_nc_go_precise[i],inherited_annots_nc_po_precise[i],inherited_annots_nc_pato_precise[i]]) for i in descriptions.keys()}
all_partial_annotations = {i:flatten([inherited_annots_nc_go_partial[i],inherited_annots_nc_po_partial[i],inherited_annots_nc_pato_partial[i]]) for i in descriptions.keys()}


# In[29]:


# Treating these sets of inherited ontology terms as tokens so that they can be used as n-grams.
processed["precise_annotations"] = {i:" ".join(annots) for i,annots in all_precise_annotations.items()}
processed["partial_annotations"] = {i:" ".join(annots) for i,annots in all_partial_annotations.items()}


# In[30]:


# Create description strings with all ontology term anntotations concatenated to the end of the descriptions.
processed["simple_plus_precise_annotations"] = {i:" ".join(flatten([text,all_precise_annotations[i]])) for i,text in processed["simple"].items()}
processed["simple_plus_partial_annotations"] = {i:" ".join(flatten([text,all_partial_annotations[i]])) for i,text in processed["simple"].items()}
processed["full_plus_precise_annotations"] = {i:" ".join(flatten([text,all_precise_annotations[i]])) for i,text in processed["full"].items()}
processed["full_plus_partial_annotations"] = {i:" ".join(flatten([text,all_partial_annotations[i]])) for i,text in processed["full"].items()}


# In[31]:


# Create ontology term annotations dictionaries for all the high confidence annotations present in the dataset.
curated_go_annotations = dataset.get_annotations_dictionary("go")
curated_po_annotations = dataset.get_annotations_dictionary("po")


# <a id="phenes"></a>
# ### Splitting the descriptions into individual phenes
# As a preprocessing step, split into a new set of descriptions that's larger. Note that phenotypes are split into phenes, and the phenes that are identical are retained as separate entries in the dataset. This makes the distance matrix calculation more needlessly expensive, because vectors need to be found for the same string more than once, but it simplifies converting the edgelist back to having IDs that reference the genes (full phenotypes) instead of the smaller phenes. If anything, that problem should be addressed in the pairwise functions, not here. (The package should handle it, not when creating input data for those methods).

# In[32]:


# Transform the processed description dictionaries back into referencing either just phenotypes or phenes.
descriptions = dataset.get_description_dictionary()
processes = list(processed.keys())
unmerged = defaultdict(dict)
for process,di in processed.items():
    unmerged[process] = {i:text for i,text in di.items() if i in phenotype_ids}
    unmerged["{}_phenes".format(process)] = {i:text for i,text in di.items() if i in phene_ids}

processed = unmerged

# Checking to make sure the size of each dictionary is as expected.
for process in processes:
    assert len(phenotype_ids) == len(processed[process].keys())
    assert len(phene_ids) == len(processed["{}_phenes".format(process)].keys())


# <a id="part_4"></a>
# # Part 4. Generating vector representations and pairwise distances matrices
# This section uses the text descriptions, preprocessed text descriptions, or ontology term annotations created or read in the previous sections to generate a vector representation for each gene and build a pairwise distance matrix for the whole dataset. Each method specified is a unique combination of a method of vectorization (bag-of-words, n-grams, document embedding model, etc) and distance metric (Euclidean, Jaccard, cosine, etc) applied to those vectors in constructing the pairwise matrix. The method of vectorization here is equivalent to feature selection, so the task is to figure out which type of vectors will encode features that are useful (n-grams, full words, only words from a certain vocabulary, etc).
# 
# <a id="methods"></a>
# ### Specifying a list of NLP methods to use

# In[70]:


doc2vec_and_word2vec_approaches = [    
    
    Method("Doc2Vec", "Wikipedia,Size=300", pw.pairwise_square_doc2vec, {"model":doc2vec_wiki_model, "ids_to_texts":descriptions, "metric":"cosine"}, spatial.distance.cosine),
    Method("Word2Vec", "Wikipedia,Size=300,Mean", pw.pairwise_square_word2vec, {"model":word2vec_model, "ids_to_texts":descriptions, "metric":"cosine", "method":"mean"}, spatial.distance.cosine),
    Method("Word2Vec", "Wikipedia,Size=300,Max", pw.pairwise_square_word2vec, {"model":word2vec_model, "ids_to_texts":descriptions, "metric":"cosine", "method":"max"}, spatial.distance.cosine),
    
    Method("Doc2Vec Phenes", "Wikipedia,Size=300", pw.pairwise_square_doc2vec, {"model":doc2vec_wiki_model, "ids_to_texts":phenes, "metric":"cosine"}, spatial.distance.cosine, tag="phenes"),
    Method("Word2Vec Phenes", "Wikipedia,Size=300,Mean", pw.pairwise_square_word2vec, {"model":word2vec_model, "ids_to_texts":phenes, "metric":"cosine", "method":"mean"}, spatial.distance.cosine, tag="phenes"),
    Method("Word2Vec Phenes", "Wikipedia,Size=300,Max", pw.pairwise_square_word2vec, {"model":word2vec_model, "ids_to_texts":phenes, "metric":"cosine", "method":"max"}, spatial.distance.cosine, tag="phenes"),
]


# In[71]:


automated_annotation_approaches = [
    
    Method("NOBLE Coder", "Precise", pw.pairwise_square_ngrams, {"ids_to_texts":processed["precise_annotations"], "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "metric":"jaccard", "tfidf":False}, spatial.distance.jaccard),
    Method("NOBLE Coder", "Partial", pw.pairwise_square_ngrams, {"ids_to_texts":processed["partial_annotations"], "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "metric":"jaccard", "tfidf":False}, spatial.distance.jaccard),
    Method("NOBLE Coder", "Precise,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":processed["precise_annotations"], "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "metric":"cosine", "tfidf":True}, spatial.distance.cosine),
    Method("NOBLE Coder", "Partial,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":processed["partial_annotations"], "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "metric":"cosine", "tfidf":True}, spatial.distance.cosine),
    
    Method("NOBLE Coder Phenes", "Precise", pw.pairwise_square_ngrams, {"ids_to_texts":processed["precise_annotations_phenes"], "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "metric":"jaccard", "tfidf":False}, spatial.distance.jaccard, tag="phenes"),
    Method("NOBLE Coder Phenes", "Partial", pw.pairwise_square_ngrams, {"ids_to_texts":processed["partial_annotations_phenes"], "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "metric":"jaccard", "tfidf":False}, spatial.distance.jaccard, tag="phenes"),
    Method("NOBLE Coder Phenes", "Precise,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":processed["precise_annotations_phenes"], "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "metric":"cosine", "tfidf":True}, spatial.distance.cosine, tag="phenes"),
    Method("NOBLE Coder Phenes", "Partial,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":processed["partial_annotations_phenes"], "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "metric":"cosine", "tfidf":True}, spatial.distance.cosine, tag="phenes"),   
]


# In[72]:


nmf_topic_modeling_approaches = [
    
    Method("Topic Models", "NMF,Full,Topics=20", pw.pairwise_square_topic_model, {"ids_to_texts":processed["full"], "metric":"cosine", "num_topics":20, "algorithm":"nmf"}, spatial.distance.cosine),
    Method("Topic Models", "NMF,Full,Topics=50", pw.pairwise_square_topic_model, {"ids_to_texts":processed["full"], "metric":"cosine", "num_topics":50, "algorithm":"nmf"}, spatial.distance.cosine),
    Method("Topic Models", "NMF,Full,Topics=100", pw.pairwise_square_topic_model, {"ids_to_texts":processed["full"], "metric":"cosine", "num_topics":100, "algorithm":"nmf"}, spatial.distance.cosine),
    Method("Topic Models", "NMF,Full,Topics=200", pw.pairwise_square_topic_model, {"ids_to_texts":processed["full"], "metric":"cosine", "num_topics":200, "algorithm":"nmf"}, spatial.distance.cosine),
    
    Method("Topic Models Phenes", "NMF,Full,Topics=20", pw.pairwise_square_topic_model, {"ids_to_texts":processed["full_phenes"], "metric":"cosine", "num_topics":20, "algorithm":"nmf"}, spatial.distance.cosine, tag="phenes"),
    Method("Topic Models Phenes", "NMF,Full,Topics=50", pw.pairwise_square_topic_model, {"ids_to_texts":processed["full_phenes"], "metric":"cosine", "num_topics":50, "algorithm":"nmf"}, spatial.distance.cosine, tag="phenes"),
    Method("Topic Models Phenes", "NMF,Full,Topics=100", pw.pairwise_square_topic_model, {"ids_to_texts":processed["full_phenes"], "metric":"cosine", "num_topics":100, "algorithm":"nmf"}, spatial.distance.cosine, tag="phenes"),
    Method("Topic Models Phenes", "NMF,Full,Topics=200", pw.pairwise_square_topic_model, {"ids_to_texts":processed["full_phenes"], "metric":"cosine", "num_topics":200, "algorithm":"nmf"}, spatial.distance.cosine, tag="phenes"),
    
]


# In[73]:


lda_topic_modeling_approaches = [

    Method("Topic Models", "LDA,Full,Topics=20", pw.pairwise_square_topic_model, {"ids_to_texts":processed["full"], "metric":"cosine", "num_topics":20, "algorithm":"lda"}, spatial.distance.cosine),
    Method("Topic Models", "LDA,Full,Topics=50", pw.pairwise_square_topic_model, {"ids_to_texts":processed["full"], "metric":"cosine", "num_topics":50, "algorithm":"lda"}, spatial.distance.cosine),
    Method("Topic Models", "LDA,Full,Topics=100", pw.pairwise_square_topic_model, {"ids_to_texts":processed["full"], "metric":"cosine", "num_topics":100, "algorithm":"lda"}, spatial.distance.cosine),
    Method("Topic Models", "LDA,Full,Topics=200", pw.pairwise_square_topic_model, {"ids_to_texts":processed["full"], "metric":"cosine", "num_topics":200, "algorithm":"lda"}, spatial.distance.cosine),

    Method("Topic Models Phenes", "LDA,Full,Topics=20", pw.pairwise_square_topic_model, {"ids_to_texts":processed["full_phenes"], "metric":"cosine", "num_topics":20, "algorithm":"lda"}, spatial.distance.cosine, tag="phenes"),
    Method("Topic Models Phenes", "LDA,Full,Topics=50", pw.pairwise_square_topic_model, {"ids_to_texts":processed["full_phenes"], "metric":"cosine", "num_topics":50, "algorithm":"lda"}, spatial.distance.cosine, tag="phenes"),
    Method("Topic Models Phenes", "LDA,Full,Topics=100", pw.pairwise_square_topic_model, {"ids_to_texts":processed["full_phenes"], "metric":"cosine", "num_topics":100, "algorithm":"lda"}, spatial.distance.cosine, tag="phenes"),
    Method("Topic Models Phenes", "LDA,Full,Topics=200", pw.pairwise_square_topic_model, {"ids_to_texts":processed["full_phenes"], "metric":"cosine", "num_topics":200, "algorithm":"lda"}, spatial.distance.cosine, tag="phenes"),

]


# In[74]:


vanilla_ngrams_approaches = [
    
    Method("N-Grams", "Full,Words,1-grams", pw.pairwise_square_ngrams, {"ids_to_texts":processed["full"], "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.cosine),
    Method("N-Grams", "Full,Words,1-grams,Binary", pw.pairwise_square_ngrams, {"ids_to_texts":processed["full"], "metric":"jaccard", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.jaccard),
    Method("N-Grams", "Full,Words,1-grams,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":processed["full"], "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine),
    Method("N-Grams", "Full,Words,1-grams,Binary,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":processed["full"], "metric":"cosine", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine),

    Method("N-Grams", "Full,Words,1-grams, 2-grams", pw.pairwise_square_ngrams, {"ids_to_texts":processed["full"], "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,2), "max_features":10000, "tfidf":False}, spatial.distance.cosine),
    Method("N-Grams", "Full,Words,1-grams,2-grams, Binary", pw.pairwise_square_ngrams, {"ids_to_texts":processed["full"], "metric":"jaccard", "binary":True, "analyzer":"word", "ngram_range":(1,2), "max_features":10000, "tfidf":False}, spatial.distance.jaccard),
    Method("N-Grams", "Full,Words,1-grams,2-grams, TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":processed["full"], "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,2), "max_features":10000, "tfidf":True}, spatial.distance.cosine),
    Method("N-Grams", "Full,Words,1-grams,2-grams, Binary,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":processed["full"], "metric":"cosine", "binary":True, "analyzer":"word", "ngram_range":(1,2), "max_features":10000, "tfidf":True}, spatial.distance.cosine),
    
    Method("N-Grams Phenes", "Full,Words,1-grams", pw.pairwise_square_ngrams, {"ids_to_texts":processed["full_phenes"], "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.cosine, tag="phenes"),
    Method("N-Grams Phenes", "Full,Words,1-grams,Binary", pw.pairwise_square_ngrams, {"ids_to_texts":processed["full_phenes"], "metric":"jaccard", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.jaccard, tag="phenes"),
    Method("N-Grams Phenes", "Full,Words,1-grams,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":processed["full_phenes"], "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine, tag="phenes"),
    Method("N-Grams Phenes", "Full,Words,1-grams,Binary,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":processed["full_phenes"], "metric":"cosine", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine, tag="phenes"),
]


# In[75]:


modified_vocab_approaches = [
    
    Method("N-Grams", "Full,Nouns,Adjectives,1-grams", pw.pairwise_square_ngrams, {"ids_to_texts":processed["nouns_adjectives_full"], "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.cosine),
    Method("N-Grams", "Linares_Pontes,Words,1-grams", pw.pairwise_square_ngrams, {"ids_to_texts":processed["linares_pontes"], "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.cosine),
    Method("N-Grams", "Full,Precise_Annotations,Words,1-grams", pw.pairwise_square_ngrams, {"ids_to_texts":processed["full_plus_precise_annotations"], "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.cosine),
    Method("N-Grams", "Full,Partial_Annotations,Words,1-grams", pw.pairwise_square_ngrams, {"ids_to_texts":processed["full_plus_partial_annotations"], "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.cosine),
    Method("N-Grams", "Full,Plant Overrepresented Tokens,1-grams", pw.pairwise_square_ngrams, {"ids_to_texts":processed["plant_overrepresented_tokens"], "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.cosine),
    Method("N-Grams", "Full,Bio Ontology Tokens,1-grams", pw.pairwise_square_ngrams, {"ids_to_texts":processed["bio_ontology_tokens"], "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.cosine),
    
    Method("N-Grams Phenes", "Full,Nouns,Adjectives,1-grams", pw.pairwise_square_ngrams, {"ids_to_texts":processed["nouns_adjectives_full_phenes"], "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.cosine, tag="phenes"),
    Method("N-Grams Phenes", "Linares_Pontes,Words,1-grams", pw.pairwise_square_ngrams, {"ids_to_texts":processed["linares_pontes_phenes"], "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.cosine, tag="phenes"),
    Method("N-Grams Phenes", "Full,Precise_Annotations,Words,1-grams", pw.pairwise_square_ngrams, {"ids_to_texts":processed["full_plus_precise_annotations_phenes"], "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.cosine, tag="phenes"),
    Method("N-Grams Phenes", "Full,Partial_Annotations,Words,1-grams", pw.pairwise_square_ngrams, {"ids_to_texts":processed["full_plus_partial_annotations_phenes"], "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.cosine, tag="phenes"),
    Method("N-Grams Phenes", "Full,Plant Overrepresented Tokens,1-grams", pw.pairwise_square_ngrams, {"ids_to_texts":processed["plant_overrepresented_tokens_phenes"], "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.cosine, tag="phenes"),
    Method("N-Grams Phenes", "Full,Bio Ontology Tokens,1-grams", pw.pairwise_square_ngrams, {"ids_to_texts":processed["bio_ontology_tokens_phenes"], "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.cosine, tag="phenes"),
]


# In[77]:


manual_annotation_approaches = [
    
    Method("GO", "None", pw.pairwise_square_annotations, {"ids_to_annotations":curated_go_annotations, "ontology":go, "metric":"jaccard", "binary":True, "analyzer":"word", "ngram_range":(1,1), "tfidf":False}, spatial.distance.jaccard),
    Method("PO", "None", pw.pairwise_square_annotations, {"ids_to_annotations":curated_po_annotations, "ontology":po, "metric":"jaccard", "binary":True, "analyzer":"word", "ngram_range":(1,1), "tfidf":False}, spatial.distance.jaccard),
]


# In[79]:


# Adding lists of approaches to the complete set to be run, this is useful when running the notebook as a script.
methods = []
if args.learning: methods.extend(doc2vec_and_word2vec_approaches)
if args.noblecoder: methods.extend(automated_annotation_approaches)
if args.nmf: methods.extend(nmf_topic_modeling_approaches)
if args.lda: methods.extend(lda_topic_modeling_approaches)
if args.vanilla: methods.extend(vanilla_ngrams_approaches)
if args.vocab: methods.extend(modified_vocab_approaches)
if args.annotations: methods.extend(manual_annotation_approaches)


# <a id="running"></a>
# ### Running all of the methods to generate distance matrices
# Notes- Instead of passing in similarity function like cosine distance that will get evaluated for every possible i,j pair of vetors that are created (this is very big when splitting by phenes), don't use a specific similarity function, but instead let the object use a KNN classifier. pass in some limit for k like 100. then the object uses some more efficient (not brute force) algorithm to set the similarity of some vector v to its 100 nearest neighbors as those 100 probabilities, and sets everything else to 0. This would need to be implemented as a matching but separate function from the get_square_matrix_from_vectors thing. And then this would need to be noted in the similarity function that was used for these in the big table of methods. This won't work because the faster (not brute force algorithms) are not for sparse vectors like n-grams, and the non-sparse embeddings aren't really the problem here because those vectors are relatively much short, even when concatenating BERT encoder layers thats only up to around length of ~1000.

# In[80]:


# Generate all the pairwise distance matrices (not in parallel).
graphs = {}
names = []
durations = []
for method in methods:
    graph,duration = function_wrapper_with_duration(function=method.function, args=method.kwargs)
    graphs[method.name_with_hyperparameters] = graph
    names.append(method.name_with_hyperparameters)
    durations.append(to_hms(duration))
    print("{:70} {}".format(method.name_with_hyperparameters,to_hms(duration)))
durations_df = pd.DataFrame({"method":names,"duration":durations})
durations_df.to_csv(os.path.join(OUTPUT_DIR,"part_4_durations.csv"), index=False)


# <a id="merging"></a>
# ### Merging all of the distance matrices into a single dataframe specifying edges
# This section also handles replacing IDs from the individual methods that are references individual phenes that are part of a larger phenotype, and replacing those IDs with IDs referencing the full phenotypes (one-to-one relationship between phenotypes and genes). In this case, the minimum distance found between any two phenes from those two phenotypes represents the distance between that pair of phenotypes.

# In[81]:


# Merging all the edgelists together.
metric_dict = {method.name_with_hyperparameters:method.metric for method in methods}
tags_dict = {method.name_with_hyperparameters:method.tag for method in methods}
names = list(graphs.keys())
edgelists = {k:v.edgelist for k,v in graphs.items()}
graphs = None

# Modify the edgelists for the methods that were using a phene split.
for name,edgelist in edgelists.items():
    # Converting phene IDs back to phenotype (gene) IDs where applicable.
    if "phene" in tags_dict[name]:
        edgelist["from"] = edgelist["from"].map(lambda x: phene_id_to_id[x])
        edgelist["to"] = edgelist["to"].map(lambda x: phene_id_to_id[x])
        edgelist = edgelist.groupby(["from","to"], as_index=False).min()
    # Making sure the edges are listed with the nodes sorted consistently.
    cond = edgelist["from"] > edgelist["to"]
    edgelist.loc[cond, ['from', 'to']] = edgelist.loc[cond, ['to', 'from']].values
    edgelists[name] = edgelist

# Do the merge step and remove self edges from the full dataframe.
df = merge_edgelists(edgelists, default_value=1.000)
df = remove_self_loops(df)
df["from"] = df["from"].astype("int64")
df["to"] = df["to"].astype("int64")
df.head(20)

# Memory cleaning up, helpful when running on the cluster with the full size dataset.
# All information from these edgelists is now in the dataframe variable. 
# TODO a better scheme overall would then be to convert each column to a sparse matrix represented by a dictionary
# maybe, but for the analysis it might be better to use the brute force approach so that no values are removed.
edgelists = None


# <a id="ensemble"></a>
# ### Combining multiple distances measurements into summarizing distance values
# The purpose of this section is to iteratively train models on subsections of the dataset using simple regression or machine learning approaches to predict a value from zero to one indicating indicating how likely is it that two genes share atleast one of the specified groups in common. The information input to these models is the distance scores provided by each method in some set of all the methods used in this notebook. The purpose is to see whether or not a function of these similarity scores specifically trained to the task of predicting common groupings is better able to used the distance metric information to report a score for this task.

# In[42]:


# Get the average distance percentile as a means of combining multiple scores.
name = "Mean"
names_to_use_for_mean = [name for name in names if not name in ["GO:None","PO:None"]]
df[name] = df[names_to_use_for_mean].rank(pct=True).mean(axis=1)
names.append(name)
df.head(20)


# In[43]:


# Normalizing all of the array representations of the graphs so they can be combined. Then this version of the arrays
# should be used by any other cells that need all of the arrays, rather than the arrays accessed from the graph
# objects. This is necessary for this analysis because some of the graph objects refer to phene datasets not
# phenotype datasets.
name_to_array = {}
ids = list(descriptions.keys())
n = len(descriptions)
id_to_array_index = {i:idx for idx,i in enumerate(ids)}
array_index_to_id = {idx:i for i,idx in id_to_array_index.items()}
for name in names:
    print(name)
    idx = list(df.columns).index(name)+1
    arr = np.ones((n, n))
    for row in df.itertuples():
        arr[id_to_array_index[row[1]]][id_to_array_index[row[2]]] = row[idx]
        arr[id_to_array_index[row[2]]][id_to_array_index[row[1]]] = row[idx]
    np.fill_diagonal(arr, 0.000) 
    name_to_array[name] = arr    


# ### Finding correlations between human and computational approaches for hand-picked phenotype pairs
# This is only meant to be run in the context of the notebook, and should never be run automatically in the script. 

# In[44]:


if NOTEBOOK:
    small_table = defaultdict(dict)
    for name in names:
        values = []
        scores = []
        for tup,score in pair_to_score.items():
            i = id_to_array_index[tup[0]]
            j = id_to_array_index[tup[1]]
            value = 1 - name_to_array[name][i,j]
            values.append(value)
            scores.append(score)
        rho,pval = spearmanr(values,scores)
        small_table[name] = {"rho":rho,"pval":pval}
    pd.DataFrame(small_table).transpose()


# <a id="part_5"></a>
# # Part 5. Biological Questions

# In[45]:


df.head(20)


# <a id="pathway_objective"></a>
# ### Using shared pathway membership (PlantCyc and KEGG) as the objective

# In[46]:


# Add a column that indicates whether or not both genes of the pair mapped to a pathway resource.
pathway_mapped_ids = set(kegg_mapped_ids+pmn_mapped_ids)
df["from_is_valid"] = df["from"].map(lambda x: x in pathway_mapped_ids)
df["to_is_valid"] = df["to"].map(lambda x: x in pathway_mapped_ids)
df["pair_is_valid"] = df["from_is_valid"]*df["to_is_valid"]


# Add a column giving the actual target output value for this biological task, with -1 for the irrelevant rows.
df["pathways"] = -1
id_to_kegg_group_ids, kegg_group_id_to_ids = kegg_groups.get_groupings_for_dataset(dataset)
id_to_pmn_group_ids, pmn_group_id_to_ids = pmn_groups.get_groupings_for_dataset(dataset)
id_to_group_ids = {i:flatten([id_to_kegg_group_ids[i],id_to_pmn_group_ids[i]]) for i in dataset.get_ids()}
df.loc[(df["pair_is_valid"]==True),"pathways"] = df[["from","to"]].apply(lambda x: len(set(id_to_group_ids[x["from"]]).intersection(set(id_to_group_ids[x["to"]])))>0, axis=1)*1
df.drop(labels=["from_is_valid","to_is_valid","pair_is_valid"], axis="columns", inplace=True)


# In[47]:


# Add a column that indicates whether or not both genes of the pair mapped to a pathway resource.
pathway_mapped_ids = set(kegg_mapped_ids)
df["from_is_valid"] = df["from"].map(lambda x: x in pathway_mapped_ids)
df["to_is_valid"] = df["to"].map(lambda x: x in pathway_mapped_ids)
df["pair_is_valid"] = df["from_is_valid"]*df["to_is_valid"]


# Add a column giving the actual target output value for this biological task, with -1 for the irrelevant rows.
df["kegg_only"] = -1
id_to_kegg_group_ids, kegg_group_id_to_ids = kegg_groups.get_groupings_for_dataset(dataset)
id_to_pmn_group_ids, pmn_group_id_to_ids = pmn_groups.get_groupings_for_dataset(dataset)
id_to_group_ids = {i:flatten([id_to_kegg_group_ids[i],id_to_pmn_group_ids[i]]) for i in dataset.get_ids()}
df.loc[(df["pair_is_valid"]==True),"kegg_only"] = df[["from","to"]].apply(lambda x: len(set(id_to_group_ids[x["from"]]).intersection(set(id_to_group_ids[x["to"]])))>0, axis=1)*1
df.drop(labels=["from_is_valid","to_is_valid","pair_is_valid"], axis="columns", inplace=True)


# In[48]:


# Add a column that indicates whether or not both genes of the pair mapped to a pathway resource.
pathway_mapped_ids = set(pmn_mapped_ids)
df["from_is_valid"] = df["from"].map(lambda x: x in pathway_mapped_ids)
df["to_is_valid"] = df["to"].map(lambda x: x in pathway_mapped_ids)
df["pair_is_valid"] = df["from_is_valid"]*df["to_is_valid"]


# Add a column giving the actual target output value for this biological task, with -1 for the irrelevant rows.
df["pmn_only"] = -1
id_to_kegg_group_ids, kegg_group_id_to_ids = kegg_groups.get_groupings_for_dataset(dataset)
id_to_pmn_group_ids, pmn_group_id_to_ids = pmn_groups.get_groupings_for_dataset(dataset)
id_to_group_ids = {i:flatten([id_to_kegg_group_ids[i],id_to_pmn_group_ids[i]]) for i in dataset.get_ids()}
df.loc[(df["pair_is_valid"]==True),"pmn_only"] = df[["from","to"]].apply(lambda x: len(set(id_to_group_ids[x["from"]]).intersection(set(id_to_group_ids[x["to"]])))>0, axis=1)*1
df.drop(labels=["from_is_valid","to_is_valid","pair_is_valid"], axis="columns", inplace=True)

df.head(20)


# <a id="subset_objective"></a>
# ### Using shared phenotype classification (Lloyd and Meinke et al., 2012) as the objective

# In[49]:


# Add a column that indicates whether or not both genes of the pair are mapped to a phenotype classification.
relevant_ids = set(subsets_mapped_ids)
df["from_is_valid"] = df["from"].map(lambda x: x in relevant_ids)
df["to_is_valid"] = df["to"].map(lambda x: x in relevant_ids)
df["pair_is_valid"] = df["from_is_valid"]*df["to_is_valid"]

# Add a column giving the actual target output value for this biological task, with -1 for the irrelevant rows.
df["subsets"] = -1
id_to_group_ids,_ = phe_subsets_groups.get_groupings_for_dataset(dataset)
df.loc[(df["pair_is_valid"]==True),"subsets"] = df[["from","to"]].apply(lambda x: len(set(id_to_group_ids[x["from"]]).intersection(set(id_to_group_ids[x["to"]])))>0, axis=1)*1
df.drop(labels=["from_is_valid","to_is_valid", "pair_is_valid"], axis="columns", inplace=True)

df.head(20)


# <a id="association_objective"></a>
# ### Using protein assocations (STRING) as the objective 

# In[50]:


# Add a column that indicates whether or not both genes of the pair are mapped to a phenotype classification.
relevant_ids = set(string_edgelist.ids)
df["from_is_valid"] = df["from"].map(lambda x: x in relevant_ids)
df["to_is_valid"] = df["to"].map(lambda x: x in relevant_ids)
df["pair_is_valid"] = df["from_is_valid"]*df["to_is_valid"]

# Add a column giving the actual target output value for this biological task, with -1 for the irrelevant rows.
df["known"] = -1
df["predicted"] = -1
df = df.merge(right=string_edgelist.df, how="left", on=["from","to"])
df["known_associations"].fillna(value=0, inplace=True)
df["predicted_associations"].fillna(value=0, inplace=True)
df.loc[(df["pair_is_valid"]==True),"known"] = df["known_associations"]
df.loc[(df["pair_is_valid"]==True),"predicted"] = df["predicted_associations"]

# Convert all the positive values from string on range 0 to arbitrary n to be equal to 1.
df.loc[df["known"] >= 1, "known"] = 1 
df.loc[df["predicted"] >= 1, "predicted"] = 1 
df.drop(labels=["from_is_valid","to_is_valid","pair_is_valid","known_associations","predicted_associations"], axis="columns", inplace=True)

df.head(20)


# <a id="ortholog_objective"></a>
# ### Using orthology between genes (PANTHER) as the objective

# In[51]:


# Add a column that indicates whether or not both genes of the pair are mapped to a phenotype classification.
relevant_ids = set(panther_edgelist.ids)
df["from_is_valid"] = df["from"].map(lambda x: x in relevant_ids)
df["to_is_valid"] = df["to"].map(lambda x: x in relevant_ids)
df["pair_is_valid"] = df["from_is_valid"]*df["to_is_valid"]

# Add a column giving the actual target output value for this biological task, with -1 for the irrelevant rows.
df["orthologs"] = -1
df = df.merge(right=panther_edgelist.df, how="left", on=["from","to"])
df["value"].fillna(value=0, inplace=True)
df.loc[(df["pair_is_valid"]==True),"orthologs"] = df["value"]
df.drop(labels=["from_is_valid","to_is_valid","pair_is_valid","value"], axis="columns", inplace=True)

df.head(20)    


# <a id="eq_sim"></a>
# ### Curator-derived similarity values from Oellrich, Walls et al., 2015

# In[52]:


# Add a column that indicates whether or not both genes of the pair are mapped to all the curation types.
relevant_ids = set(ow_edgelist.ids)
df["from_is_valid"] = df["from"].map(lambda x: x in relevant_ids)
df["to_is_valid"] = df["to"].map(lambda x: x in relevant_ids)
df["pair_is_valid"] = df["from_is_valid"]*df["to_is_valid"]

# Add a column giving the actual target output value for this biological task, with -1 for the irrelevant rows.
df["eqs_distance"] = -1
df = df.merge(right=ow_edgelist.df, how="left", on=["from","to"])
df["value"].fillna(value=0, inplace=True)
df.loc[(df["pair_is_valid"]==True),"eqs_distance"] = 1-df["value"]
df.drop(labels=["from_is_valid","to_is_valid","pair_is_valid","value"], axis="columns", inplace=True)

# Also, add the curated EQ approach to the list of column names that reference approaches to be evaluated.
names.append("eqs_distance")

df.head(20)   


# <a id="curated"></a>
# ### Checking whether gene pairs are considered curated or not

# In[82]:


# Add a column that indicates whether or not both genes of the pair are mapped to all the curation types.
relevant_ids = set(ids_with_all_annotations)
df["from_is_valid"] = df["from"].map(lambda x: x in relevant_ids)
df["to_is_valid"] = df["to"].map(lambda x: x in relevant_ids)
df["curated"] = df["from_is_valid"]*df["to_is_valid"]
df.drop(labels=["from_is_valid","to_is_valid"], axis="columns", inplace=True)
df.head(10)   


# <a id="species"></a>
# ### Checking whether gene pairs are intraspecies or not

# In[83]:


species_dict = dataset.get_species_dictionary()
df["same"] = df[["from","to"]].apply(lambda x: species_dict[x["from"]]==species_dict[x["to"]],axis=1)
df.head(10)


# ### Checking to make sure that the number of genes and pairs matches what is expected at this point

# In[55]:


# Defining a nested dictionary with shape dict[curated][question][species][approach][metric] --> value.
curated = [True,False]
species = ["intra","inter","both"]
question = ["subsets", "known", "predicted", "pathways", "orthologs"]
tables = defaultdict(dict)
for c,q in itertools.product(curated,question): 
    tables[c][q] = defaultdict(dict)
for c,q,s in itertools.product(curated,question,species): 
    tables[c][q][s] = defaultdict(dict)


# <a id="n_values"></a>
# ### What are the value of *n* for each type of iteration through a subset of the dataset?

# In[56]:


subset_idx_lists = []
subset_properties = []
table_lists = defaultdict(list)
for c,q,s in itertools.product(curated,question,species):
    
    # Remembering what the properties for this particular subset are.
    subset_properties.append((c,q,s))
    
    # Subsetting the dataframe to the rows (gene pairs) that are relevant for this particular biological question.
    subset = df[df[q] != -1]
    if c:
        subset = subset[subset["curated"] == True]
        
        
    # Subsetting the dataframe to the rows (gene pairs) where both genes are from the same or different species.
    if s == "intra":
        subset = subset[subset["same"] == True]
    elif s == "inter":
        subset = subset[subset["same"] == False]
        
    subset_idx_lists.append(subset.index.to_list())
    
    # Adding values to the table that are specific to this biological question.
    counts = Counter(subset[q].values)
    
    table_lists["question"].append(q.lower())
    table_lists["curated"].append(str(c).lower())
    table_lists["species"].append(s.lower())
    table_lists["num_genes"].append(len(set(subset["to"].values).union(set(subset["from"].values))))
    table_lists["positive"].append(counts[1])
    table_lists["negative"].append(counts[0])
    #table_lists["class_ratio"].append("{:0.4f}".format(counts[1]/counts[0]))

pairs_table = pd.DataFrame(table_lists)  
pairs_table["num_pairs"] = pairs_table["positive"]+pairs_table["negative"]
pairs_table["positive_fraction"] = pairs_table["positive"] / pairs_table["num_pairs"]
pairs_table["negative_fraction"] = pairs_table["negative"] / pairs_table["num_pairs"]
pairs_table.to_csv(os.path.join(OUTPUT_DIR,"part_5_biological_question_n_values.csv"), index=False)
pairs_table


# <a id="objective_similarities"></a>
# ### How similar are the different biological objectives to each other?

# In[57]:


# Looking more at the distributions of target values for each of the biological questions.
from scipy.spatial.distance import jaccard
row_tuples = []
for q1,q2 in itertools.combinations(question, 2):
    q1_subset = df[df[q1] != -1]
    q2_subset = df[df[q2] != -1]
    overlap_subset  = q1_subset[q1_subset[q2] != -1]
    q1_num_pairs = q1_subset.shape[0]
    q2_num_pairs = q2_subset.shape[0]
    overlap_size = overlap_subset.shape[0]
    overlap_sim = 1-jaccard(overlap_subset[q1].values, overlap_subset[q2].values)
    row_tuples.append((q1, q2, q1_num_pairs, q2_num_pairs, overlap_size, overlap_sim))
question_overlaps_table = pd.DataFrame(row_tuples)
question_overlaps_table.columns = ["question_1", "question_2", "num_pairs_1", "num_pairs_2", "num_overlap", "sim_overlap"]
question_overlaps_table.sort_values(by="sim_overlap", ascending=False, inplace=True)
question_overlaps_table.reset_index(inplace=True, drop=True)
question_overlaps_table.to_csv(os.path.join(OUTPUT_DIR,"part_5_biological_question_overlaps.csv"), index=False)
question_overlaps_table


# <a id="part_6"></a>
# # Part 6. Results

# <a id="ks"></a>
# ### Do the edges joining genes that share a group, pathway, or interaction come from a different distribution?
# The purpose of this section is to visualize kernel estimates for the distributions of distance or similarity scores generated by each of the methods tested for measuring semantic similarity or generating vector representations of the phenotype descriptions. Ideally, better methods should show better separation betwene the distributions for distance values between two genes involved in a common specified group or two genes that are not. Additionally, a statistical test is used to check whether these two distributions are significantly different from each other or not, although this is a less informative measure than the other tests used in subsequent sections, because it does not address how useful these differences in the distributions actually are for making predictions about group membership.

# In[58]:


for properties,idxs in zip(subset_properties, subset_idx_lists):
    
    # Remember the properties for this subset being looked at, and subset the dataframe accordingly.
    c,q,s = properties
    
    # Don't look at the inter-species and intra-species edges except for pathways, otherwise irrelevant.
    if (s != "both") and (q != "pathways"):
        continue
    
    # Only look at gene pairs where both are relevant to the given biological question.
    subset = df.loc[idxs]
        
    # Check that this subsetting leaves a valid dataset with both positive and negatives samples.
    class_values = pd.unique(subset[q].values)
    if not (len(class_values)==2 and 0 in class_values and 1 in class_values):
        continue
    
    # Use Kolmogorov-Smirnov test to see if edges between genes that share a group come from a distinct distribution.
    ppi_pos_dict = {name:(subset[subset[q] > 0.00][name].values) for name in names}
    ppi_neg_dict = {name:(subset[subset[q] == 0.00][name].values) for name in names}
    for name in names:
        stat,p = ks_2samp(ppi_pos_dict[name],ppi_neg_dict[name])
        pos_mean = np.average(ppi_pos_dict[name])
        neg_mean = np.average(ppi_neg_dict[name])
        pos_n = len(ppi_pos_dict[name])
        neg_n = len(ppi_neg_dict[name])
        
        tables[c][q][s][name].update({"mean_1":pos_mean, "mean_0":neg_mean, "n_1":pos_n, "n_0":neg_n})
        tables[c][q][s][name].update({"ks":stat, "ks_pval":p})

    # Show the kernel estimates for each distribution of weights for each method.
    #num_plots, plots_per_row, row_width, row_height = (len(names), 4, 14, 3)
    #fig,axs = plt.subplots(math.ceil(num_plots/plots_per_row), plots_per_row, squeeze=False)
    #for name,ax in zip(names,axs.flatten()):
    #    ax.set_title(name)
    #    ax.set_xlabel("value")
    #    ax.set_ylabel("density")
    #    sns.kdeplot(ppi_pos_dict[name], color="black", shade=False, alpha=1.0, ax=ax)
    #    sns.kdeplot(ppi_neg_dict[name], color="black", shade=True, alpha=0.1, ax=ax) 
    #fig.set_size_inches(row_width, row_height*math.ceil(num_plots/plots_per_row))
    #fig.tight_layout()
    #fig.savefig(os.path.join(OUTPUT_DIR,"part_6_kernel_density.png"),dpi=400)
    #plt.close()
    
print("done with all kolmogorov–smirnov tests")


# <a id="within"></a>
# ### Looking at within-group or within-pathway distances in each graph
# The purpose of this section is to determine which methods generated graphs which tightly group genes which share common pathways or group membership with one another. In order to compare across different methods where the distance value distributions are different, the mean distance values for each group for each method are convereted to percentile scores. Lower percentile scores indicate that the average distance value between any two genes that belong to that group is lower than most of the distance values in the entire distribution for that method.

# In[59]:


# What are the different groupings we are interested in for these mean within-group distance tables?
grouping_objects = [kegg_groups, pmn_groups, phe_subsets_groups]
grouping_names = ["kegg_only","pmn_only","subsets"]
for (groups,q) in zip(grouping_objects,grouping_names):

    # Only look at gene pairs where both are relevant to the given biological question.
    subset = df[df[q] != -1]
    
    # The grouping dictionaries for this particular biological question.    
    id_to_group_ids, group_id_to_ids = groups.get_groupings_for_dataset(dataset)

    # Get all the average within-group distance values for each approach.
    group_ids = list(group_id_to_ids.keys())
    graph = IndexedGraph(subset)
    within_percentiles_dict = defaultdict(lambda: defaultdict(list))
    all_weights_dict = {}
    for name in names:
        for group in group_ids:
            within_ids = group_id_to_ids[group]
            within_pairs = [(i,j) for i,j in itertools.permutations(within_ids,2)]
            mean_weight = np.mean((graph.get_values(within_pairs, kind=name)))
            within_percentiles_dict[name][group] = stats.percentileofscore(subset[name].values, mean_weight, kind="rank")

    # Generating a dataframe of percentiles of the mean in-group distance scores.
    within_dist_data = pd.DataFrame(within_percentiles_dict)
    within_dist_data = within_dist_data.dropna(axis=0, inplace=False)
    within_dist_data = within_dist_data.round(4)

    # Adding relevant information to this dataframe and saving.
    # Defining mean_group_rank: the average of the individual rank given to this pathway by each approach.
    # Defining mean_avg_pair_percentile: the average across all approaches of the average distance percentile for each gene pair.
    within_dist_data["mean_group_rank"] = within_dist_data.rank().mean(axis=1)
    within_dist_data["mean_avg_pair_percentile"] = within_dist_data.mean(axis=1)
    within_dist_data.sort_values(by="mean_avg_pair_percentile", inplace=True)
    within_dist_data.reset_index(inplace=True)
    within_dist_data["group_id"] = within_dist_data["index"]
    within_dist_data["full_name"] = within_dist_data["group_id"].apply(lambda x: groups.get_long_name(x))
    within_dist_data["n"] = within_dist_data["group_id"].apply(lambda x: len(group_id_to_ids[x]))
    within_dist_data = within_dist_data[flatten(["group_id","full_name","n","mean_avg_pair_percentile","mean_group_rank",names])]
    within_dist_data.to_csv(os.path.join(OUTPUT_DIR,"part_5_{}_within_distances.csv".format(q)), index=False)
    within_dist_data.head(5)

print("done generating mean within-group distance values")


# <a id="auc"></a>
# ### Predicting whether two genes belong to the same group, pathway, or share an interaction
# The purpose of this section is to see if whether or not two genes share atleast one common pathway can be predicted from the distance scores assigned using analysis of text similarity. The evaluation of predictability is done by reporting a precision and recall curve for each method, as well as remembering the area under the curve, and ratio between the area under the curve and the baseline (expected area when guessing randomly) for each method.

# In[60]:


def bootstrap(fraction, num_iterations, y_true, y_prob):
    # Run the desired number of bootstrap iterations over the full population of predictions and return st devs.
    scores = pd.DataFrame([bootstrap_iteration(fraction, y_true, y_prob) for i in range(num_iterations)])
    standard_deviations = {
        "f_1_max_std": np.std(scores["f_1_max"].values),
        "f_2_max_std": np.std(scores["f_2_max"].values),
        "f_point5_max_std": np.std(scores["f_point5_max"].values)}
    return(standard_deviations)


def bootstrap_iteration(fraction, y_true, y_prob):
    assert len(y_true) == len(y_prob)
    # Subset the total population of predictions using the provided fraction.
    num_predictions = len(y_true)
    bootstrapping_fraction = fraction
    num_to_retain = int(np.ceil(num_predictions*bootstrapping_fraction))
    idx = np.random.choice(np.arange(num_predictions), num_to_retain, replace=False)
    y_true_sample = y_true[idx]
    y_prob_sample = y_prob[idx]
    
    # Calculate any desired metrics using just that subset.
    n_pos, n_neg = Counter(y_true_sample)[1], Counter(y_true_sample)[0]
    precision, recall, thresholds = precision_recall_curve(y_true_sample, y_prob_sample)
    baseline = Counter(y_true_sample)[1]/len(y_true_sample) 
    area = auc(recall, precision)
    auc_to_baseline_auc_ratio = area/baseline
    
    # Find the maximum Fß score for different values of ß.  
    f_beta = lambda pr,re,beta: [((1+beta**2)*p*r)/((((beta**2)*p)+r)) for p,r in zip(pr,re)]
    f_1_scores = f_beta(precision,recall,beta=1)
    f_2_scores = f_beta(precision,recall,beta=2)
    f_point5_scores = f_beta(precision,recall,beta=0.5)
    
    # Create a dictionary of those metric values to return.
    scores={"f_1_max":np.nanmax(f_1_scores),"f_2_max":np.nanmax(f_2_scores),"f_point5_max":np.nanmax(f_point5_scores)}
    return(scores)


# In[61]:


for properties,idxs in zip(subset_properties, subset_idx_lists):
    
    # Remember the properties for this subset being looked at, and subset the dataframe accordingly.
    c,q,s = properties
    
    # Don't look at the inter-species and intra-species edges except for pathways, otherwise irrelevant.
    if (s != "both") and (q != "pathways"):
        continue
    
    # Create a subset of the dataframe that contains only the gene pairs for this question.
    subset = df.loc[idxs]

    # Check that this subsetting leaves a valid dataset with both positive and negatives samples.
    class_values = pd.unique(subset[q].values)
    if not (len(class_values)==2 and 0 in class_values and 1 in class_values):
        continue



    y_true_dict = {name:subset[q].values for name in names}       #just added .values here...
    y_prob_dict = {name:(1 - subset[name].values) for name in names}
    num_plots, plots_per_row, row_width, row_height = (len(names), 4, 14, 3)
    fig,axs = plt.subplots(math.ceil(num_plots/plots_per_row), plots_per_row, squeeze=False)
    for name,ax in zip(names, axs.flatten()):

        # Obtaining the values and metrics.
        y_true, y_prob = y_true_dict[name], y_prob_dict[name]
        n_pos, n_neg = Counter(y_true)[1], Counter(y_true)[0]
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        baseline_auc = Counter(y_true)[1]/len(y_true) 
        area = auc(recall, precision)
        auc_to_baseline_auc_ratio = area/baseline_auc
        # The baseline F1 max has a precision of the ratio of positives to all samples and a recall of 1.
        # This is because a random classifier achieves that precision at all recall values, so recall is maximized to
        # find the maximum F1 value that can be expected due to random chance.
        baseline_f1_max = (2*baseline_auc*1)/(baseline_auc+1)

        #TABLE[name].update({"auc":area,"ratio":auc_to_baseline_auc_ratio, "baseline":baseline_f1_max, })
        tables[c][q][s][name].update({"auc":area,"ratio":auc_to_baseline_auc_ratio, "baseline":baseline_f1_max, })


        # Find the maximum Fß score for different values of ß.  
        f_beta = lambda pr,re,beta: [((1+beta**2)*p*r)/((((beta**2)*p)+r)) for p,r in zip(pr,re)]
        f_1_scores = f_beta(precision,recall,beta=1)
        f_2_scores = f_beta(precision,recall,beta=2)
        f_point5_scores = f_beta(precision,recall,beta=0.5)
        f_1_max, f_1_std = np.nanmax(f_1_scores), np.std(f_1_scores)
        f_2_max, f_2_std = np.nanmax(f_2_scores), np.std(f_2_scores)
        f_point5_max, f_point5_std = np.nanmax(f_point5_scores), np.std(f_point5_scores)

        # Find the standard deviation of each metric when subsampling the dataset of predictions for each method.
        bootstrap_fraction = 0.5
        bootstrap_iterations = 2
        bootstrapped_std_dict = bootstrap(bootstrap_fraction, bootstrap_iterations, y_true, y_prob)

        tables[c][q][s][name].update({"f1_max":f_1_max, "f5_max":f_point5_max, "f2_max":f_2_max})
        #TABLE[name].update({"f1_std":f_1_std, "f5_std":f_point5_std, "f2_std":f_2_std})
        tables[c][q][s][name].update({"f1_std":bootstrapped_std_dict["f_1_max_std"], 
                            "f5_std":bootstrapped_std_dict["f_point5_max_std"], 
                            "f2_std":bootstrapped_std_dict["f_2_max_std"]}) 

        # Producing the precision recall curve.
        #step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
        #ax.step(recall, precision, color='black', alpha=0.2, where='post')
        #ax.fill_between(recall, precision, alpha=0.7, color='black', **step_kwargs)
        #ax.axhline(baseline_auc, linestyle="--", color="lightgray")
        #ax.set_xlabel('Recall')
        #ax.set_ylabel('Precision')
        #ax.set_ylim([0.0, 1.05])
        #ax.set_xlim([0.0, 1.0])
        #ax.set_title("PR {0} (Baseline={1:0.3f})".format(name, baseline_auc))

    #fig.set_size_inches(row_width, row_height*math.ceil(num_plots/plots_per_row))
    #fig.tight_layout()
    #fig.savefig(os.path.join(OUTPUT_DIR,"part_5_prcurve_shared.png"),dpi=400)
    plt.close()

print("done with finding precision and recall values for each approach")


# <a id="y"></a>
# ### Are genes in the same group or pathway ranked higher with respect to individual nodes?
# This is a way of statistically seeing if for some value k, the graph ranks more edges from some particular gene to any other gene that it has a true protein-protein interaction with higher or equal to rank k, than we would expect due to random chance. This way of looking at the problem helps to be less ambiguous than the previous methods, because it gets at the core of how this would actually be used. In other words, we don't really care how much true information we're missing as long as we're still able to pick up some new useful information by building these networks, so even though we could be missing a lot, what's going on at the very top of the results? These results should be comparable to very strictly thresholding the network and saying that the remaining edges are our guesses at interactions. This is comparable to just looking at the far left-hand side of the precision recall curves, but just quantifies it slightly differently.

# In[62]:


if NOTEBOOK:
    
    # When the edgelist is generated above, only the lower triangle of the pairwise matrix is retained for edges in the 
    # graph. This means that in terms of the indices of each node, only the (i,j) node is listed in the edge list where
    # i is less than j. This makes sense because the graph that's specified is assumed to already be undirected. However
    # in order to be able to easily subset the edgelist by a single column to obtain rows that correspond to all edges
    # connected to a particular node, this method will double the number of rows to include both (i,j) and (j,i) edges.
    df = make_undirected(df)

    # What's the number of functional partners ranked k or higher in terms of phenotypic description similarity for 
    # each gene? Also figure out the maximum possible number of functional partners that could be theoretically
    # recovered in this dataset if recovered means being ranked as k or higher here.
    k = 10      # The threshold of interest for gene ranks.
    n = 100     # Number of Monte Carlo simulation iterations to complete.
    df[list(names)] = df.groupby("from")[list(names)].rank()
    ys = df[df["shared"]==1][list(names)].apply(lambda s: len([x for x in s if x<=k]))
    ymax = sum(df.groupby("from")["shared"].apply(lambda s: min(len([x for x in s if x==1]),k)))

    # Monte Carlo simulation to see what the probability is of achieving each y-value by just randomly pulling k 
    # edges for each gene rather than taking the top k ones that the similarity methods specifies when ranking.
    ysims = [sum(df.groupby("from")["shared"].apply(lambda s: len([x for x in s.sample(k) if x>0.00]))) for i in range(n)]
    for name in names:
        pvalue = len([ysim for ysim in ysims if ysim>=ys[name]])/float(n)
        TABLE[name].update({"y":ys[name], "y_max":ymax, "y_ratio":ys[name]/ymax, "y_pval":pvalue})


# <a id="mean"></a>
# ### Predicting biochemical pathway or group membership based on mean vectors
# This section looks at how well the biochemical pathways that a particular gene is a member of can be predicted based on the similarity between the vector representation of the phenotype descriptions for that gene and the average vector for all the vector representations of phenotypes asociated with genes that belong to that particular pathway. In calculating the average vector for a given biochemical pathway, the vector corresponding to the gene that is currently being classified is not accounted for, to avoid overestimating the performance by including information about the ground truth during classification. This leads to missing information in the case of biochemical pathways that have only one member. This can be accounted for by only limiting the overall dataset to only include genes that belong to pathways that have atleast two genes mapped to them, and only including those pathways, or by removing the missing values before calculating the performance metrics below.

# In[63]:


# Get the list of methods to look at, and a mapping between each method and the correct similarity metric to apply.
# vector_dicts = {k:v.vector_dictionary for k,v in graphs.items()}
# names = list(vector_dicts.keys())
# group_id_to_ids = groups.get_group_id_to_ids_dict(dataset.get_gene_dictionary())
# valid_group_ids = [group for group,id_list in group_id_to_ids.items() if len(id_list)>1]
# valid_ids = [i for i in dataset.get_ids() if len(set(valid_group_ids).intersection(set(id_to_group_ids[i])))>0]
# pred_dict = defaultdict(lambda: defaultdict(dict))
# true_dict = defaultdict(lambda: defaultdict(dict))
# for name in names:
#     for group in valid_group_ids:
#         ids = group_id_to_ids[group]
#         for identifier in valid_ids:
#             # What's the mean vector of this group, without this particular one that we're trying to classify.
#             vectors = np.array([vector_dicts[name][some_id] for some_id in ids if not some_id==identifier])
#             mean_vector = vectors.mean(axis=0)
#             this_vector = vector_dicts[name][identifier]
#             pred_dict[name][identifier][group] = 1-metric_dict[name](mean_vector, this_vector)
#             true_dict[name][identifier][group] = (identifier in group_id_to_ids[group])*1                


# In[64]:


# num_plots, plots_per_row, row_width, row_height = (len(names), 4, 14, 3)
# fig,axs = plt.subplots(math.ceil(num_plots/plots_per_row), plots_per_row, squeeze=False)
# for name,ax in zip(names, axs.flatten()):
#     
#     # Obtaining the values and metrics.
#     y_true = pd.DataFrame(true_dict[name]).as_matrix().flatten()
#     y_prob = pd.DataFrame(pred_dict[name]).as_matrix().flatten()
#     n_pos, n_neg = Counter(y_true)[1], Counter(y_true)[0]
#     precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
#     baseline = Counter(y_true)[1]/len(y_true) 
#     area = auc(recall, precision)
#     auc_to_baseline_auc_ratio = area/baseline
#     TABLE[name].update({"mean_auc":area, "mean_baseline":baseline, "mean_ratio":auc_to_baseline_auc_ratio})
# 
#     # Producing the precision recall curve.
#     step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
#     ax.step(recall, precision, color='black', alpha=0.2, where='post')
#     ax.fill_between(recall, precision, alpha=0.7, color='black', **step_kwargs)
#     ax.axhline(baseline, linestyle="--", color="lightgray")
#     ax.set_xlabel('Recall')
#     ax.set_ylabel('Precision')
#     ax.set_ylim([0.0, 1.05])
#     ax.set_xlim([0.0, 1.0])
#     ax.set_title("PR {0} (Baseline={1:0.3f})".format(name[:10], baseline))
#     
# fig.set_size_inches(row_width, row_height*math.ceil(num_plots/plots_per_row))
# fig.tight_layout()
# fig.savefig(os.path.join(OUTPUT_DIR,"part_6_prcurve_mean_classifier.png"),dpi=400)
# plt.close()


# ### Predicting biochemical pathway membership based on mean similarity values
# This section looks at how well the biochemical pathways that a particular gene is a member of can be predicted based on the average similarity between the vector representationt of the phenotype descriptions for that gene and each of the vector representations for other phenotypes associated with genes that belong to that particular pathway. In calculating the average similarity to other genes from a given biochemical pathway, the gene that is currently being classified is not accounted for, to avoid overestimating the performance by including information about the ground truth during classification. This leads to missing information in the case of biochemical pathways that have only one member. This can be accounted for by only limiting the overall dataset to only include genes that belong to pathways that have atleast two genes mapped to them, and only including those pathways, or by removing the missing values before calculating the performance metrics below.

# ### Predicting biochemical pathway or group membership with KNN classifier
# This section looks at how well the group(s) or biochemical pathway(s) that a particular gene belongs to can be predicted based on a KNN classifier generated using every other gene. For this section, only the groups or pathways which contain more than one gene, and the genes mapped to those groups or pathways, are of interest. This is because for other genes, if we consider them then it will be true that that gene belongs to that group in the target vector, but the KNN classifier could never predict this because when that gene is held out, nothing could provide a vote for that group, because there are zero genes available to be members of the K nearest neighbors.

# <a id="output"></a>
# ### Summarizing the results for this notebook
# Write a large table of results to an output file. Columns are generally metrics and rows are generally methods.

# In[65]:


result_dfs = []
for s,c,q in itertools.product(species,curated,question):
    if (s != "both") and (q != "pathways"):
        continue
    TABLE = tables[c][q][s]
    results = pd.DataFrame(TABLE).transpose()
    columns = flatten(["Species", "Objective","Curated","Hyperparameters","Group","Order",results.columns])
    results["Hyperparameters"] = ""
    results["Group"] = "NLP"
    results["Order"] = np.arange(results.shape[0])
    results["Species"] = s.lower()
    results["Objective"] = q.lower()
    results["Curated"] = str(c).lower()
    results = results[columns]
    results.reset_index(inplace=True)
    results = results.rename({"index":"Method"}, axis="columns")
    hyperparam_sep = ":"
    results["Hyperparameters"] = results["Method"].map(lambda x: x.split(hyperparam_sep)[1] if hyperparam_sep in x else "None")
    results["Method"] = results["Method"].map(lambda x: x.split(hyperparam_sep)[0])
    result_dfs.append(results)

results = pd.concat(result_dfs)
results.reset_index(inplace=True, drop=True)
results.to_csv(os.path.join(OUTPUT_DIR,"part_5_full_table.csv"), index=False)
results.head(20)


# In[66]:


# Make another version of the table that is more useful for looking at one particular metric or value.
metric_of_interest = "f1_max"
reshaped_results = results[["Method","Hyperparameters","Order"]].drop_duplicates()
for c,q,s in itertools.product(curated,question,species):
    if (s != "both") and (q != "pathways"):
        continue
    c_label = {True:"curated",False:"all"}[c]
    col_name = "{}_{}_{}".format(s,c_label,q)
    reshaped_results[col_name] = reshaped_results["Order"].map(lambda x: results.loc[(results["Order"]==x) & (results["Curated"]==str(c).lower()) & (results["Objective"]==q.lower()) & (results["Species"]==s.lower()), metric_of_interest])
    reshaped_results[col_name] = reshaped_results[col_name].map(lambda x: None if len(x)==0 else x.values[0])
reshaped_results.to_csv(os.path.join(OUTPUT_DIR,"part_5_partial_table_reshaped.csv"), index=False)
reshaped_results


# <a id="part_6"></a>
# # Part 6. Clustering Analysis
# The purpose of this section is to look at different ways that the embeddings obtained for the dataset of phenotype descriptions can be used to cluster or organize the genes to which those phenotypes are mapped into subgroups or representations. These approaches include generating topic models from the data, and doing agglomerative clustering to find clusters to which each gene belongs.

# In[67]:


# Objects initially created in previous sections that are used by this section.
groups = phe_subsets_groups
id_to_group_ids, group_id_to_ids = groups.get_groupings_for_dataset(dataset)


# <a id="topic_modeling"></a>
# ### Approach 1: Topic modeling based on n-grams with a reduced vocabulary
# Topic modelling learns a set of word probability distributions from the dataset of text descriptions, which represent distinct topics which are present in the dataset. Each text description can then be represented as a discrete probability distribution over the learned topics based on the probability that a given piece of text belongs to each particular topics. This is a form of data reduction because a high dimensionsal bag-of-words can be represented as a vector of *k* probabilities where *k* is the number of topics. The main advantages of topic modelling over clustering is that topic modelling provides soft classifications that can be additionally interpreted, rather than hard classifications into a single cluster. Topic models are also explainable, because the word probability distributions for that topic can be used to determine which words are most representative of any given topic. One problem with topic modelling is that is uses the n-grams embeddings to semantic similarity between different words is not accounted for. To help alleviate this, this section uses implementations of some existing algorithms to compress the vocabulary as a preprocessing step based on word distance matrices generated using word embeddings.
# 
# Topic models define topics present in a dataset of texts as word or n-gram probability distributions. These models represent each instance of text then as being composed of or generated as as mixture of these topics. The vector for each text that indicates which fraction of that text is generated by a each topic is of length *n* where *n* is the number of topics, and can be used as a reduced dimensionality of the text, with a much smaller vector length than the n-grams embedding itself. Therefore we can build a topic model of the data with 100 topics for example in order to then represent each description in the dataset as a a vector of length 100. This section constructs topic models from the n-gram representations of the dataset and selects different values for the number of topics in order to find a value that works well during the grid search over the training dataset.

# In[68]:


# What to use as the list of texts from which to actually generate the topic model, one of the processed ones above.
texts = [processed["simple"][i] for i in subsets_mapped_ids]
texts = [processed["full"][i] for i in subsets_mapped_ids]

# TODO make a version of this section that produces results for each combination of these parameters.
algorithms = ["lda","nmf"]
preprocessing = ["simple","full"]
datatype = ["phenotypes","phenes"]

# Basic parameters for this problem that are currently used.
number_of_topics = 42
seed = 0


# In[ ]:


# Creating and fitting the topic model, either NFM or LDA or something like that.
vectorizer = TfidfVectorizer(max_features=10000, stop_words="english", max_df=0.95, min_df=2, lowercase=True)
features = vectorizer.fit_transform(texts)
cls = NMF(n_components=number_of_topics, random_state=seed)
cls.fit(features)

# Function for retrieving the topic vectors for a list of text descriptions.
def get_topic_embeddings(texts, model, vectorizer):
    ngrams_vectors = vectorizer.transform(texts).toarray()
    topic_vectors = model.transform(ngrams_vectors)
    return(topic_vectors)

# Create the dataframe containing the average score assigned to each topic for the genes from each subset.
group_to_topic_vector = {}
for group_id,ids in group_id_to_ids.items():
    texts = [descriptions[i] for i in ids]
    topic_vectors = get_topic_embeddings(texts, cls, vectorizer)
    mean_topic_vector = np.mean(topic_vectors, axis=0)
    group_to_topic_vector[group_id] = mean_topic_vector
    
# Turning that matrix of weights into a dataframe so it can be worked with.
tm_df = pd.DataFrame(group_to_topic_vector)

# Changing the order of the Lloyd, Meinke phenotype subsets to match other figures and tables for consistency.
lmtm_df = pd.read_csv(lloyd_function_hierarchy_path)    
columns_in_order = [col for col in lmtm_df["Subset Symbol"].values if col in tm_df.columns]
columns_in_order.reverse()
assert len(columns_in_order) == number_of_topics
tm_df = tm_df[columns_in_order]
    
# Reordering so consistency with the curated subsets can be checked by looking at the diagonal.
tm_df["idxmax"] = tm_df.idxmax(axis = 1)
tm_df["idxmax"] = tm_df["idxmax"].apply(lambda x: tm_df.columns.get_loc(x))
tm_df = tm_df.sort_values(by="idxmax")
tm_df.drop(columns=["idxmax"], inplace=True)

# Saving a version of this dataframe this is indexed by topic integers and subset strings, before makings topics a column instead.
topic_subset_similarity_df = tm_df
tm_df = tm_df.reset_index(drop=False).rename({"index":"topic"},axis=1).reset_index(drop=False).rename({"index":"order"},axis=1)


# In[ ]:


topic_subset_similarity_df.head(10)


# In[ ]:


tm_df.head(10)


# In[ ]:


# Describing what the most representative tokens for each topic in the model are.
num_top_words = 5
map_top_words = {}
feature_names = vectorizer.get_feature_names()
for i,topic_vec in enumerate(cls.components_):
    top_words = []
    print(i,end=": ")
    for fid in topic_vec.argsort()[-1:-num_top_words-1:-1]:
        word = feature_names[fid]
        # The next line is applicable if words in the topic model are actually a function of the words in the texts.
        #word = " ".join(unreduce[word])
        top_words.append(word)
        print(word, end=" ")  
    map_top_words[i] = top_words
    print()


# In[ ]:


# Create a column that specifies what the top tokens for each topic are.
tm_df["tokens"] = tm_df["topic"].map(lambda x: "|".join(map_top_words[x]))

# Move that column to the left for readability before writing to the file.
tokens_col = tm_df.pop("tokens")
tm_df.insert(2, "tokens", tokens_col)

# Renaming the topics to be in order, to be more helpful when preparing figures that are more intuitive.
tm_df["topic_renumbered"] = tm_df["order"].values[::-1]+1
topic_renumbered_col = tm_df.pop("topic_renumbered")
tm_df.insert(2, "topic_renumbered", topic_renumbered_col)

# Remembering a mapping between the topics, their order, and what the renumbered names are.
topic_order_map = {t:i for t,i in zip(tm_df["topic"].values, tm_df["order"].values)}
topic_renumbered_map = {t:i for t,i in zip(tm_df["topic"].values, tm_df["topic_renumbered"].values)}

# Saving this version of the subset and topic similarity data to a file.
tm_df.to_csv(os.path.join(OUTPUT_DIR,"part_6_topic_modeling_matrix.csv"), index=False)
tm_df.head(10)


# In[ ]:


# Producing a version of the previous table that is useful for producing line drawings representing these results.
tm_lines_dict = defaultdict(list)


# Remembering the order of the Lloyd, Meinke phenotype subsets to match other figures for consistency.
lmtm_df = pd.read_csv(lloyd_function_hierarchy_path)   
subset_to_class_map = {s:c for s,c in zip(lmtm_df["Subset Symbol"].values, lmtm_df["Class Name"].values)}
subset_to_desc_map = {s:c for s,c in zip(lmtm_df["Subset Symbol"].values, lmtm_df["Subset Name and Description "].values)}
subset_abbrevs_in_order = [col for col in lmtm_df["Subset Symbol"].values if col in tm_df.columns]
subset_abbrevs_in_order.reverse()
subset_order_map = {subset_abbrev:i for i,subset_abbrev in enumerate(subset_abbrevs_in_order)}



# Producing the line entries that represent connections between the subsets and topics.
line_number = 0
topic_int_list = list(topic_subset_similarity_df.columns)
subset_str_list = list(topic_subset_similarity_df.index)
for subset_abbrev, topic_int in itertools.product(topic_int_list,subset_str_list):
    
    # The weight of the line, extracted from the similarity matrix between subsets and topics built previously.
    weight = topic_subset_similarity_df.loc[topic_int,subset_abbrev]
    
    # The strings that should be used to represent classes, subsets, and topics in a figure or plot.
    subset_str = "{} ({})".format(subset_abbrev, subset_to_desc_map[subset_abbrev].lower())
    tm_lines_dict["subset_str"].extend([subset_str,subset_str])
    tm_lines_dict["class_str"].extend([subset_to_class_map[subset_abbrev],subset_to_class_map[subset_abbrev]])
    topic_str = "Topic {}: ({})".format(topic_renumbered_map[topic_int], "|".join(map_top_words[topic_int]))
    tm_lines_dict["topic_str"].extend([topic_str,topic_str])
    
    # Which line is this, they all have individual numbers so that each line can be its own group in a ggplot object.
    tm_lines_dict["line_number"].extend([line_number,line_number])
    tm_lines_dict["weight"].extend([weight,weight])
    
    # Where should the line start and stop? The horizontal values are arbitrary and just have to match.
    # The vertical values are determined by which subset and topic are being connected to each other.
    tm_lines_dict["x"].extend([0,10])
    tm_lines_dict["y"].extend([subset_order_map[subset_abbrev],topic_order_map[topic_int]])
    
    line_number = line_number+1
    
tm_lines_df = pd.DataFrame(tm_lines_dict)
tm_lines_df.to_csv(os.path.join(OUTPUT_DIR,"part_6_topic_modeling_lines.csv"), index=False)
tm_lines_df.head(50)


# <a id="clustering"></a>
# ### Approach 2: Agglomerative clustering and comparison to predefined groups
# This clustering approach uses agglomerative clustering to cluster the genes into a fixed number of clusters based off the distances between their embedding representations using all of the above methods. Clustering into a fixed number of clusters allows for clustering into a similar number of groups as a present in some existing grouping of the data, such as phenotype categories or biochemical pathways, and then determining if the clusters obtained are at all similar to the groupings that already exist. Agglomerative clustering is used here in order to use an arbitrary predefined distance matrix, in this case the matrix being used is the mean distance percentiles from each of the different approaches.

# In[ ]:


# Generate the numpy array where values are mean distance percentiles between all the methods.
mean_pct_array = name_to_array["Mean"]
to_id = array_index_to_id

# Do agglomerative clustering based on that distance matrix.
number_of_clusters = 42
ac = AgglomerativeClustering(n_clusters=number_of_clusters, linkage="complete", affinity="precomputed")
clustering = ac.fit(mean_pct_array)
id_to_cluster = {}
cluster_to_ids = defaultdict(list)
for idx,c in enumerate(clustering.labels_):
    id_to_cluster[to_id[idx]] = c
    cluster_to_ids[c].append(to_id[idx])


# In[ ]:


# Create the dataframe containing the average score assigned to each topic for the genes from each subset.
group_to_cluster_vector = {}
for group_id,ids in group_id_to_ids.items():
    
    mean_cluster_vector = np.zeros(number_of_clusters)
    for i in ids:
        cluster = id_to_cluster[i]
        mean_cluster_vector[cluster] = mean_cluster_vector[cluster]+1
    mean_cluster_vector = mean_cluster_vector/mean_cluster_vector.sum(axis=0,keepdims=1)
    group_to_cluster_vector[group_id] = mean_cluster_vector
    
ac_df = pd.DataFrame(group_to_cluster_vector)

# Changing the order of the Lloyd, Meinke phenotype subsets to match other figures for consistency.
filename = "../data/group_related_files/lloyd/lloyd_function_hierarchy_irb_cleaned.csv"
lmtm_df = pd.read_csv(filename)    
columns_in_order = [col for col in lmtm_df["Subset Symbol"].values if col in ac_df.columns]
ac_df = ac_df[columns_in_order]

# Reordering so consistency with the curated subsets can be checked by looking at the diagonal.
ac_df["idxmax"] = ac_df.idxmax(axis = 1)
ac_df["idxmax"] = ac_df["idxmax"].apply(lambda x: ac_df.columns.get_loc(x))
ac_df = ac_df.sort_values(by="idxmax")
ac_df.drop(columns=["idxmax"], inplace=True)
ac_df = ac_df.reset_index(drop=False).rename({"index":"cluster"},axis=1).reset_index(drop=False).rename({"index":"order"},axis=1)
ac_df.to_csv(os.path.join(OUTPUT_DIR,"part_6_agglomerative_clustering.csv"), index=False)
ac_df


# ### Approach 3: Agglomerative clustering and sillhouette scores for each NLP method

# In[90]:


if NOTEBOOK:
    from sklearn.metrics.cluster import silhouette_score
    # Note that homogeneity scores don't fit for evaluating how close the clustering is to pathway membership, etc.
    # This is because genes can be assigned to more than one pathway, metric would have to be changed to account for this.
    # So all this section does is determines which values of n_clusters provide good clustering results for each matrix.
    n_clusters_silhouette_scores = defaultdict(dict)
    min_n_clusters = 20
    max_n_clusters = 80
    step_size = 4
    number_of_clusters = np.arange(min_n_clusters, max_n_clusters, step_size)
    for n in number_of_clusters:
        for name in names:
            distance_matrix = name_to_array[name]
            ac = AgglomerativeClustering(n_clusters=n, linkage="complete", affinity="precomputed")
            clustering = ac.fit(distance_matrix)
            sil_score = silhouette_score(distance_matrix, clustering.labels_, metric="precomputed")
            n_clusters_silhouette_scores[name][n] = sil_score
    sil_df = pd.DataFrame(n_clusters_silhouette_scores).reset_index(drop=False).rename({"index":"n"},axis="columns")
    sil_df.to_csv(os.path.join(OUTPUT_DIR,"part_6_silhouette_scores_by_n.csv"), index=False)
    sil_df.head(10)


# <a id="phenologs"></a>
# ### Approach 4: Looking for phenolog relationships between clusters and OMIM disease phenotypes
# This section produces a table of values that provides a score for the a particular pair of a cluster found for this dataset of plant genes and a disease phenotype. Currently the value indicates the fraction of the plant genes in that cluster that have orthologs associated with that disease phenotype. This should be replaced or supplemented with a p-value for evaluating the significance of this value given the distribution of genes and their mappings to all of the disease phenotypes. All the rows from the input dataframe containing the PantherDB and OMIM information where the ID from this dataset is not known or the mapping to a phenotype was unsuccessful are removed at this step, fix this if the metric for evaluating cluster to phenotype phenolog mappings need this information.

# In[ ]:


# Read in the dataframe mapping plant genes --> human orthologs --> disease phenotypes.
omim_df = pd.read_csv(panther_to_omim_filename)
# Add a column that indicates which ID in the dataset those plant genes refer to, for mapping to phenotypes.
name_to_id = dataset.get_name_to_id_dictionary()
omim_df["id"] = omim_df["gene_identifier"].map(lambda x: name_to_id.get(x,None))
omim_df = omim_df.dropna(subset=["id","phenotype_mim_name"], inplace=False)
omim_df["phenotype_mim_name"] = omim_df["phenotype_mim_name"].astype(str)
omim_df["compressed_phenotype_mim_name"] = omim_df["phenotype_mim_name"].map(lambda x: x.split(",")[0])
omim_df["id"] = omim_df["id"].astype("int64")
omim_df["phenotype_mim_number"] = omim_df["phenotype_mim_number"].astype("int64")
# Generate mappings between the IDs in this dataset and disease phenotypes or orthologous genes.
id_to_mim_phenotype_names = defaultdict(list)
for i,p in zip(omim_df["id"].values,omim_df["compressed_phenotype_mim_name"].values):
    id_to_mim_phenotype_names[i].append(p)
id_to_human_gene_symbols = defaultdict(list)
for i,s in zip(omim_df["id"].values,omim_df["human_ortholog_gene_symbol"].values):
    id_to_human_gene_symbols[i].append(s)
omim_df.head(5)


# In[ ]:


# How many genes in our dataset map to orthologs that map to the same OMIM phenotype?
omim_df.groupby("compressed_phenotype_mim_name").size()


# In[ ]:


phenolog_x_dict = defaultdict(dict)
phenolog_p_dict = defaultdict(dict)
candidate_genes_dict = defaultdict(dict)
phenotypes = pd.unique(omim_df["compressed_phenotype_mim_name"].values)
clusters = list(cluster_to_ids.keys())
for cluster,phenotype in itertools.product(clusters,phenotypes):
    
    # What are the candidate genes predicted if this phenolog pairing is real?
    ids = cluster_to_ids[cluster]
    candidate_genes_dict[cluster][phenotype] = list(set(flatten([id_to_human_gene_symbols[i] for i in ids if phenotype not in id_to_mim_phenotype_names.get(i,[])])))

    # What is the p-value for this phenolog pairing?
    # The size of the population (genes in the dataset).
    M = len(id_to_cluster.keys())
    # The number of elements we draw without replacement (genes in the cluster).
    N = len(cluster_to_ids[cluster])     
    # The number of available successes in the population (genes that map to orthologs that map to this phenotype).
    n = len([i for i in id_to_cluster.keys() if phenotype in id_to_mim_phenotype_names.get(i,[])])
    # The number of successes drawn (genes in this cluster that map to orthologs that map to this phenotype).
    x = list(set(flatten([id_to_mim_phenotype_names.get(i,[]) for i in ids]))).count(phenotype)
    prob = 1-hypergeom.cdf(x-1, M, n, N) # Equivalent to prob = 1-sum([hypergeom.pmf(x_i, M, n, N) for x_i in range(0,x)])
    phenolog_x_dict[cluster][phenotype] = x
    phenolog_p_dict[cluster][phenotype] = prob
    

# Convert the dictionary to a table of values with cluster and phenotype as the rows and columns.
phenolog_matrix = pd.DataFrame(phenolog_x_dict)        
phenolog_matrix.head(5)


# In[ ]:


# Produce a melted version of the phenolog matrix sorted by value and including predicted candidate genes.
phenolog_matrix_reset = phenolog_matrix.reset_index(drop=False).rename({"index":"omim_phenotype_name"}, axis="columns")
phenolog_df = pd.melt(phenolog_matrix_reset, id_vars=["omim_phenotype_name"], value_vars=phenolog_matrix.columns[1:], var_name="cluster", value_name="x")
# What other information should be present in this melted phenologs matrix?
phenolog_df["size"] = phenolog_df["cluster"].map(lambda x: len(cluster_to_ids[x]))
phenolog_df["candidate_gene_symbols"] = np.vectorize(lambda x,y: concatenate_with_bar_delim(*candidate_genes_dict[x][y]))(phenolog_df["cluster"], phenolog_df["omim_phenotype_name"])
phenolog_df["p_value"] = np.vectorize(lambda x,y: phenolog_p_dict[x][y])(phenolog_df["cluster"], phenolog_df["omim_phenotype_name"])
phenolog_df["p_adjusted"] = multipletests(phenolog_df["p_value"].values, method='bonferroni')[1]
phenolog_df.sort_values(by=["p_value"], inplace=True, ascending=True)
phenolog_df = phenolog_df[["omim_phenotype_name", "cluster", "size", "x", "p_value", "p_adjusted", "candidate_gene_symbols"]]
phenolog_df.to_csv(os.path.join(OUTPUT_DIR,"part_6_phenologs.csv"), index=False)
phenolog_df.head(30)

