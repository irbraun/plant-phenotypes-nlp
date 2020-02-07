#!/usr/bin/env python
# coding: utf-8

# # Table of Contents
# 
# - [Introduction](#introduction)
# 
# - [Links of Interest](#links)
# 
# - [Part 1. Loading and Filtering Data](#paths)
#     - [Setting input and output paths](#paths)
#     - [Reading in datasets of phenotype descriptions](#read_this_data_)
#     - [Reading in datasets of groupings](#read_other_data) (biochemical pathways, functional groups, etc.)
#     - [Relating the datasets](#relating)
#     - [Filtering the datasets](#filtering)
#     
# - [Part 2. NLP Models](#word2vec_doc2vec)
#     - [Word2Vec and Doc2Vec](#word2vec_doc2vec)
#     - [BERT and BioBERT](#bert_biobert)
#     - [Loading models](#load_models)
# 
# - [Part 3. NLP Choices]()
#     - [Preprocessing descriptions](#preprocessing)
#     - [POS Tagging](#pos_tagging)
#     - [Reducing vocabulary size](#vocab)
#     - [Annotating with biological ontologies](#annotation)
#     - [Splitting into phene descriptions](#phenes)
#     
# - [Part 4. Generating Vectors and Distance Matrices](#matrix)
#     - [Defining methods to use](#methods)
#     - [Running all methods](#running)
#     - [Merging distances into an edgelist](#merging)
#     - [Adding edge information](#merging)
# 
# - [Part 5. Clustering Analysis]()
#     - [Topic modeling](#topic_modeling)
#     - [Agglomerative clustering](#clustering)
#     - [Phenologs for OMIM disease phenotypes](#phenologs)
#     
# - [Part 6. Supervised Tasks](#supervised)
#     - [Distributions of distance values](#ks)
#     - [Within-group distance values](#within)
#     - [Predictions and AUC for shared pathways or interactions](#auc)
#     - [Tests for querying to recover related genes](#y)
#     - [Producing output summary table](#output)

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
import multiprocessing as mp
from collections import Counter, defaultdict
from inspect import signature
from scipy.stats import ks_2samp, hypergeom
from sklearn.metrics import precision_recall_curve, f1_score, auc
from sklearn.model_selection import train_test_split, KFold
from scipy import spatial, stats
from statsmodels.sandbox.stats.multicomp import multipletests
from nltk.corpus import brown
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.neighbors import KNeighborsClassifier
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from gensim.parsing.preprocessing import strip_non_alphanum, stem_text, preprocess_string, remove_stopwords
from gensim.utils import simple_preprocess
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.cluster import AgglomerativeClustering

sys.path.append("../../oats")
from oats.utils.utils import save_to_pickle, load_from_pickle, merge_list_dicts, flatten, to_hms
from oats.datasets.dataset import Dataset
from oats.datasets.groupings import Groupings
from oats.annotation.ontology import Ontology
from oats.datasets.string import String
from oats.datasets.edges import Edges
from oats.annotation.annotation import annotate_using_noble_coder
from oats.graphs import pairwise as pw
from oats.graphs.editing import merge_edgelists, make_undirected, remove_self_loops, subset_edgelist_with_ids
from oats.graphs.indexed import IndexedGraph
from oats.graphs.weighting import train_logistic_regression_model, apply_logistic_regression_model
from oats.graphs.weighting import train_random_forest_model, apply_random_forest_model
from oats.nlp.vocabulary import get_overrepresented_tokens, get_vocabulary_from_tokens
from oats.nlp.vocabulary import reduce_vocabulary_connected_components, reduce_vocabulary_linares_pontes
from oats.utils.utils import function_wrapper_with_duration
from oats.nlp.preprocess import concatenate_with_bar_delim

from _utils import Method

mpl.rcParams["figure.dpi"] = 400
warnings.simplefilter('ignore')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
nltk.download('punkt', quiet=True)
nltk.download('brown', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)


# # Part 1. Loading and Filtering Data
# <a id="paths"></a>
# ### Setting up the input and output paths and summarizing output table
# This section defines some constants which are used for creating a uniquely named directory to contain all the outputs from running this instance of this notebook. The naming scheme is based on the time that the notebook is run. The other constants are used for specifying information in the output table about what the topic was for this notebook when it was run, such as looking at KEGG biochemical pathways or STRING protein-protein interaction data some other type of gene function grouping or hierarchy. These values are arbitrary and are just for keeping better notes about what the output of the notebook corresponds to. All the input and output file paths for loading datasets or models are also contained within this cell, so that if anything is moved the directories and file names should only have to be changed at this point and nowhere else further into the notebook. If additional files are added to the notebook cells they should be put here as well.

# In[2]:


# The summarizing output dictionary has the shape TABLE[method][metric] --> value.
NOTEBOOK_TAGS = {"kegg":False, "pmn":False, "classes":False, "subsets":True}
TOPIC = "Biochemical Pathways"
DATA = "Filtered"
TABLE = defaultdict(dict)
OUTPUT_DIR = os.path.join("../outputs",datetime.datetime.now().strftime('%m_%d_%Y_h%Hm%Ms%S'))
os.mkdir(OUTPUT_DIR)


# In[3]:


dataset_filename = "../data/pickles/text_plus_annotations_dataset.pickle"                            # The full dataset pickle.
kegg_pathways_filename = "../data/pickles/kegg_pathways.pickle"                                      # The pathway groupings from KEGG.
pmn_pathways_filename = "../data/pickles/pmn_pathways.pickle"                                        # The pahway groupings from Plant Metabolic Network.
lloyd_subsets_filename = "../data/pickles/lloyd_subsets.pickle"                                      # The functional subsets defined by Lloyd and Meinke (2012).
lloyd_classes_filename = "../data/pickles/lloyd_classes.pickle"                                      # The functional classes defined by Lloyd and Meinke (2012).
background_corpus_filename = "../data/corpus_related_files/untagged_text_corpora/background.txt"     # Text file with background content.
phenotypes_corpus_filename = "../data/corpus_related_files/untagged_text_corpora/phenotypes_all.txt" # Text file with specific content.
doc2vec_pubmed_filename = "../gensim/pubmed_dbow/doc2vec_2.bin"                                      # File holding saved Doc2Vec model.
doc2vec_wikipedia_filename = "../gensim/enwiki_dbow/doc2vec.bin"                                     # File holding saved Doc2Vec model.
word2vec_model_filename = "../gensim/wiki_sg/word2vec.bin"                                           # File holding saved Word2Vec model.
ontology_filename = "../ontologies/mo.obo"                                                           # Ontology file in OBO format.
noblecoder_jarfile_path = "../lib/NobleCoder-1.0.jar"                                                # Jar for NOBLE Coder tool.
biobert_pmc_path = "../gensim/biobert_v1.0_pmc/pytorch_model"                                        # Path for PyTorch BioBERT model.
biobert_pubmed_path = "../gensim/biobert_v1.0_pubmed/pytorch_model"                                  # Path for PyTorch BioBERT model.
biobert_pubmed_pmc_path = "../gensim/biobert_v1.0_pubmed_pmc/pytorch_model"                          # Path for PyTorch BioBERT model.
panther_to_omim_filename = "../data/orthology_related_files/pantherdb_omim_df.csv"                   # File with mappings to human orthologs and disease phenotypes.


# <a id="read_this_data"></a>
# ### Reading in the dataset of genes and their associated phenotype descriptions and annotations

# In[4]:


dataset = load_from_pickle(dataset_filename)
dataset.describe()
dataset.filter_by_species("ath","zma","sly")
dataset.filter_has_description()
dataset.filter_has_annotation()
dataset.describe()
dataset.filter_has_annotation("GO")
dataset.filter_has_annotation("PO")
#dataset.filter_random_k(200)
dataset.describe()
dataset.to_pandas().head(10)


# <a id="read_other_data"></a>
# ### Reading in the dataset of groupings from KEGG, Plant Metabolic Network, and other hierarchies.

# In[5]:


groupings_filename = ""
groupings_filename = kegg_pathways_filename if NOTEBOOK_TAGS["kegg"] else groupings_filename
groupings_filename = pmn_pathways_filename if NOTEBOOK_TAGS["pmn"] else groupings_filename
groupings_filename = lloyd_subsets_filename if NOTEBOOK_TAGS["subsets"] else groupings_filename
groupings_filename = lloyd_classes_filename if NOTEBOOK_TAGS["classes"] else groupings_filename
groups = load_from_pickle(groupings_filename)
id_to_group_ids = groups.get_id_to_group_ids_dict(dataset.get_gene_dictionary())
group_id_to_ids = groups.get_group_id_to_ids_dict(dataset.get_gene_dictionary())
group_mapped_ids = [k for (k,v) in id_to_group_ids.items() if len(v)>0]
groups.to_csv(os.path.join(OUTPUT_DIR,"part_1_groupings.csv"))
groups.to_pandas().head(10)


# <a id="relating"></a>
# ### Relating the dataset of genes to the dataset of groupings or categories
# This section generates tables that indicate how the genes present in the dataset were mapped to the defined pathways or groups. This includes a summary table that indicates how many genes by species were succcessfully mapped to atleast one pathway or group, as well as a more detailed table describing how many genes from each species were mapped to each particular pathway or group.

# In[6]:


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
table.to_csv(os.path.join(OUTPUT_DIR,"part_1_mappings_summary.csv"), index=False)

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
table.to_csv(os.path.join(OUTPUT_DIR,"part_1_mappings_by_group.csv"), index=False)


# <a id="filtering"></a>
# ### Option 1: Filtering the dataset based on presence in the curated Oellrich, Walls et al. (2015) dataset

# In[7]:


# Filter the dataset based on whether or not the genes were in the curated dataset.
# This is similar to filtering based on protein interaction data because the dataset is a list of edge values.
pppn_edgelist_path = "../data/supplemental_files_oellrich_walls/13007_2015_53_MOESM9_ESM.txt"
pppn_edgelist = Edges(dataset.get_name_to_id_dictionary(), pppn_edgelist_path)
dataset.filter_with_ids(pppn_edgelist.ids)
dataset.describe()


# ### Option 2: Filtering the dataset based on protein-protein interactions
# This is done to only include genes (and the corresponding phenotype descriptions and annotations) which are useful for the current analysis. In this case we want to only retain genes that are mentioned atleast one time in the STRING database for a given species. If a gene is not mentioned at all in STRING, there is no information available for whether or not it interacts with any other proteins in the dataset so choose to not include it in the analysis. Only genes that have atleast one true positive are included because these are the only ones for which the missing information (negatives) is meaningful. This should be run instead of the subsequent cell, or the other way around, based on whether or not protein-protein interactions is the prediction goal for the current analysis.

# # Filter the dataset based on whether or not the genes were successfully mapped to an interaction.
# # Reduce size of the dataset by removing genes not mentioned in the STRING.
# naming_file = "../data/group_related_files/string/all_organisms.name_2_string.tsv"
# interaction_files = [
#     "../data/group_related_files/string/3702.protein.links.detailed.v11.0.txt", # Arabidopsis thaliana
#     "../data/group_related_files/string/4577.protein.links.detailed.v11.0.txt", # maize
#     "../data/group_related_files/string/4530.protein.links.detailed.v11.0.txt", # tomato 
#     "../data/group_related_files/string/4081.protein.links.detailed.v11.0.txt", # medicago
#     "../data/group_related_files/string/3880.protein.links.detailed.v11.0.txt", # rice 
#     "../data/group_related_files/string/3847.protein.links.detailed.v11.0.txt", # soybean
# ]
# genes = dataset.get_gene_dictionary()
# string_data = String(genes, naming_file, *interaction_files)
# dataset.filter_with_ids(string_data.ids)
# dataset.describe()

# ### Option 3: Filtering the dataset based on membership in pathways or phenotype category
# This is done to only include genes (and the corresponding phenotype descriptions and annotations) which are useful for the current analysis. In this case we want to only retain genes that are mapped to atleast one pathway in whatever the source of pathway membership we are using is (KEGG, Plant Metabolic Network, etc). This is because for these genes, it will be impossible to correctly predict their pathway membership, and we have no evidence that they belong or do not belong in certain pathways so they can not be identified as being true or false negatives in any case.

# In[8]:


# Filter based on succcessful mappings to groups or pathways.
dataset.filter_with_ids(group_mapped_ids)
dataset.describe()
# Get the mappings in each direction again now that the dataset has been subset.
id_to_group_ids = groups.get_id_to_group_ids_dict(dataset.get_gene_dictionary())
group_id_to_ids = groups.get_group_id_to_ids_dict(dataset.get_gene_dictionary())


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

# In[9]:


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


# # Part 3. NLP Choices
# 
# <a id="preprocessing"></a>
# ### Preprocessing text descriptions
# The preprocessing methods applied to the phenotype descriptions are a choice which impacts the subsequent vectorization and similarity methods which construct the pairwise distance matrix from each of these descriptions. The preprocessing methods that make sense are also highly dependent on the vectorization method or embedding method that is to be applied. For example, stemming (which is part of the full proprocessing done below using the Gensim preprocessing function) is useful for the n-grams and bag-of-words methods but not for the document embeddings methods which need each token to be in the vocabulary that was constructed and used when the model was trained. For this reason, embedding methods with pretrained models where the vocabulary is fixed should have a lighter degree of preprocessing not involving stemming or lemmatization but should involve things like removal of non-alphanumerics and normalizing case. 

# In[10]:


# Obtain a mapping between IDs and the raw text descriptions associated with that ID from the dataset.
descriptions = dataset.get_description_dictionary()

# Preprocessing of the text descriptions. Different methods are necessary for different approaches.
descriptions_full_preprocessing = {i:" ".join(preprocess_string(d)) for i,d in descriptions.items()}
descriptions_simple_preprocessing = {i:" ".join(simple_preprocess(d)) for i,d in descriptions.items()}
descriptions_no_stopwords = {i:remove_stopwords(d) for i,d in descriptions.items()}


# <a id="pos_tagging"></a>
# ### POS tagging the phenotype descriptions for nouns and adjectives
# Note that preprocessing of the descriptions should be done after part-of-speech tagging, because tokens that are removed during preprocessing before n-gram analysis contain information that the parser needs to accurately call parts-of-speech. This step should be done on the raw descriptions and then the resulting bags of words can be subset using additional preprocesssing steps before input in one of the vectorization methods.

# In[11]:


get_pos_tokens = lambda text,pos: " ".join([t[0] for t in nltk.pos_tag(word_tokenize(text)) if t[1].lower()==pos.lower()])
descriptions_noun_only =  {i:get_pos_tokens(d,"NN") for i,d in descriptions.items()}
descriptions_noun_only_full_preprocessing = {i:" ".join(preprocess_string(d)) for i,d in descriptions_noun_only.items()}
descriptions_noun_only_simple_preprocessing = {i:" ".join(simple_preprocess(d)) for i,d in descriptions_noun_only.items()}
descriptions_adj_only =  {i:get_pos_tokens(d,"JJ") for i,d in descriptions.items()}
descriptions_adj_only_full_preprocessing = {i:" ".join(preprocess_string(d)) for i,d in descriptions_adj_only.items()}
descriptions_adj_only_simple_preprocessing = {i:" ".join(simple_preprocess(d)) for i,d in descriptions_adj_only.items()}
descriptions_noun_adj = {i:"{} {}".format(descriptions_noun_only[i],descriptions_adj_only[i]) for i in descriptions.keys()}
descriptions_noun_adj_full_preprocessing = {i:"{} {}".format(descriptions_noun_only_full_preprocessing[i],descriptions_adj_only_full_preprocessing[i]) for i in descriptions.keys()}
descriptions_noun_adj_simple_preprocessing = {i:"{} {}".format(descriptions_noun_only_simple_preprocessing[i],descriptions_adj_only_simple_preprocessing[i]) for i in descriptions.keys()}


# <a id="vocab"></a>
# ### Reducing the vocabulary size using a word distance matrix
# These approaches for reducing the vocabulary size of the dataset work by replacing multiple words that occur throughout the dataset of descriptions with an identical word that is representative of this larger group of words. The total number of unique words across all descriptions is therefore reduced, and when observing n-gram overlaps between vector representations of these descriptions, overlaps will now occur between descriptions that included different but similar words. These methods work by actually generating versions of these descriptions that have the word replacements present. The returned objects for these methods are the revised description dictionary, a dictionary mapping tokens in the full vocabulary to tokens in the reduced vocabulary, and a dictionary mapping tokens in the reduced vocabulary to a list of tokens in the full vocabulary.

# In[12]:


# Reducing the size of the vocabulary for descriptions treated with simple preprocessing.
tokens = list(set([w for w in flatten(d.split() for d in descriptions_simple_preprocessing.values())]))
tokens_dict = {i:w for i,w in enumerate(tokens)}
graph = pw.pairwise_square_word2vec(word2vec_model, tokens_dict, "cosine")

# Make sure that the tokens list is in the same order as the indices representing each word in the distance matrix.
# This is only trivial here because the IDs used are ordered integers 0 to n, but this might not always be the case.
distance_matrix = graph.array
tokens = [tokens_dict[graph.index_to_id[index]] for index in np.arange(distance_matrix.shape[0])]
n = 3
threshold = 0.2
descriptions_linares_pontes, reduce_lp, unreduce_lp = reduce_vocabulary_linares_pontes(descriptions_simple_preprocessing, tokens, distance_matrix, n)
descriptions_connected_components, reduce_cc, unreduce_cc = reduce_vocabulary_connected_components(descriptions_simple_preprocessing, tokens, distance_matrix, threshold)


# ### Reducing vocabulary size based on identifying important words
# These approcahes for reducing the vocabulary size of the dataset work by identifying which words in the descriptions are likely to be the most important for identifying differences between the phenotypes and meaning of the descriptions. One approach is to determine which words occur at a higher rate in text of interest such as articles about plant phenotypes as compared to their rates in more general texts such as a corpus of news articles. These approaches do not create modified versions of the descriptions but rather provide vocabulary objects that can be passed to the sklearn vectorizer or constructors.

# In[13]:


# Constructing a vocabulary by looking at what words are overrepresented in domain specific text.
background_corpus = open(background_corpus_filename,"r").read()
phenotypes_corpus = open(phenotypes_corpus_filename,"r").read()
tokens = get_overrepresented_tokens(phenotypes_corpus, background_corpus, max_features=5000)
vocabulary_from_text = get_vocabulary_from_tokens(tokens)

# Constructing a vocabulary by assuming all words present in a given ontology are important.
ontology = Ontology(ontology_filename)
vocabulary_from_ontology = get_vocabulary_from_tokens(ontology.get_tokens())


# <a id="annotation"></a>
# ### Annotating descriptions with ontology terms
# This section generates dictionaries that map gene IDs from the dataset to lists of strings, where those strings are ontology term IDs. How the term IDs are found for each gene entry with its corresponding phenotype description depends on the cell below. Firstly, the terms are found by using the NOBLE Coder annotation tool through these wrapper functions to identify the terms by looking for instances of the term's label or synonyms in the actual text of the phenotype descriptions. Secondly, the next cell just draws the terms directly from the dataset itself. In this case, these are high-confidence annotations done by curators for a comparison against what can be accomplished through computational analysis of the text.

# In[14]:


# Run the ontology term annotators over the raw input text descriptions. NOBLE-Coder handles simple issues like case
# normalization so preprocessed descriptions are not used for this step.
ontology = Ontology(ontology_filename)
annotations_noblecoder_precise = annotate_using_noble_coder(descriptions, noblecoder_jarfile_path, "mo", precise=1)
annotations_noblecoder_partial = annotate_using_noble_coder(descriptions, noblecoder_jarfile_path, "mo", precise=0)


# In[15]:


# Get the ID to term list annotation dictionaries for each ontology in the dataset.
annotations = dataset.get_annotations_dictionary()
go_annotations = {k:[term for term in v if term[0:2]=="GO"] for k,v in annotations.items()}
po_annotations = {k:[term for term in v if term[0:2]=="PO"] for k,v in annotations.items()}


# <a id="phenes"></a>
# ### Splitting the descriptions into individual phenes
# As a preprocessing step, split into a new set of descriptions that's larger. Note that phenotypes are split into phenes, and the phenes that are identical are retained as separate entries in the dataset. This makes the distance matrix calculation more needlessly expensive, because vectors need to be found for the same string more than once, but it simplifies converting the edgelist back to having IDs that reference the genes (full phenotypes) instead of the smaller phenes. If anything, that problem should be addressed in the pairwise functions, not here. (The package should handle it, not when creating input data for those methods).

# In[16]:


# Create a dictionary of phene descriptions and a dictionary to convert back to the phenotype/gene IDs.
phenes = {}
phene_id_to_id = {}
phene_id = 0
for i,phene_list in {i:sent_tokenize(d) for i,d in descriptions.items()}.items():
    for phene in phene_list:
        phenes[phene_id] = phene
        phene_id_to_id[phene_id] = i
        phene_id = phene_id+1
        
# Repeating the reprocessing options for the individual phenes instead of the full phenotype descriptions.
phenes_full_preprocessing = {i:" ".join(preprocess_string(d)) for i,d in phenes.items()}
phenes_simple_preprocessing = {i:" ".join(simple_preprocess(d)) for i,d in phenes.items()}
phenes_no_stopwords = {i:remove_stopwords(d) for i,d in phenes.items()}
get_pos_tokens = lambda text,pos: " ".join([t[0] for t in nltk.pos_tag(word_tokenize(text)) if t[1].lower()==pos.lower()])
phenes_noun_only =  {i:get_pos_tokens(d,"NN") for i,d in phenes.items()}
phenes_noun_only_full_preprocessing = {i:" ".join(preprocess_string(d)) for i,d in phenes_noun_only.items()}
phenes_noun_only_simple_preprocessing = {i:" ".join(simple_preprocess(d)) for i,d in phenes_noun_only.items()}
phenes_adj_only =  {i:get_pos_tokens(d,"JJ") for i,d in phenes.items()}
phenes_adj_only_full_preprocessing = {i:" ".join(preprocess_string(d)) for i,d in phenes_adj_only.items()}
phenes_adj_only_simple_preprocessing = {i:" ".join(simple_preprocess(d)) for i,d in phenes_adj_only.items()}
phenes_noun_adj = {i:"{} {}".format(phenes_noun_only[i],phenes_adj_only[i]) for i in phenes.keys()}
phenes_noun_adj_full_preprocessing = {i:"{} {}".format(phenes_noun_only_full_preprocessing[i],phenes_adj_only_full_preprocessing[i]) for i in phenes.keys()}
phenes_noun_adj_simple_preprocessing = {i:"{} {}".format(phenes_noun_only_simple_preprocessing[i],phenes_adj_only_simple_preprocessing[i]) for i in phenes.keys()}

# Repeating the vocbulary reduction step using the individual phenes instead of full phenotype descriptions.
tokens = list(set([w for w in flatten(d.split() for d in phenes_simple_preprocessing.values())]))
tokens_dict = {i:w for i,w in enumerate(tokens)}
graph = pw.pairwise_square_word2vec(word2vec_model, tokens_dict, "cosine")
distance_matrix = graph.array
tokens = [tokens_dict[graph.index_to_id[index]] for index in np.arange(distance_matrix.shape[0])]
n = 3
threshold = 0.2
phenes_linares_pontes, reduce_lp, unreduce_lp = reduce_vocabulary_linares_pontes(phenes_simple_preprocessing, tokens, distance_matrix, n)
phenes_connected_components, reduce_cc, unreduce_cc = reduce_vocabulary_connected_components(phenes_simple_preprocessing, tokens, distance_matrix, threshold)


# <a id="matrix"></a>
# # Part 4. Generating vector representations and pairwise distances matrices
# This section uses the text descriptions, preprocessed text descriptions, or ontology term annotations created or read in the previous sections to generate a vector representation for each gene and build a pairwise distance matrix for the whole dataset. Each method specified is a unique combination of a method of vectorization (bag-of-words, n-grams, document embedding model, etc) and distance metric (Euclidean, Jaccard, cosine, etc) applied to those vectors in constructing the pairwise matrix. The method of vectorization here is equivalent to feature selection, so the task is to figure out which type of vectors will encode features that are useful (n-grams, full words, only words from a certain vocabulary, etc).
# 
# <a id="methods"></a>
# ### Specifying a list of NLP methods to use
# Something here if needed.

# In[17]:


methods = [

    # Full phenotype descriptions


    # Methods that use neural networks to generate embeddings.
    Method("Doc2Vec", "Wikipedia,Size=300", pw.pairwise_square_doc2vec, {"model":doc2vec_wiki_model, "ids_to_texts":descriptions, "metric":"cosine"}, spatial.distance.cosine),
    Method("Doc2Vec", "PubMed,Size=100", pw.pairwise_square_doc2vec, {"model":doc2vec_pubmed_model, "ids_to_texts":descriptions, "metric":"cosine"}, spatial.distance.cosine),
    Method("Word2Vec", "Wikipedia,Size=300,Mean", pw.pairwise_square_word2vec, {"model":word2vec_model, "ids_to_texts":descriptions, "metric":"cosine", "method":"mean"}, spatial.distance.cosine),
    Method("Word2Vec", "Wikipedia,Size=300,Max", pw.pairwise_square_word2vec, {"model":word2vec_model, "ids_to_texts":descriptions, "metric":"cosine", "method":"max"}, spatial.distance.cosine),

    #Method("BERT", "Base:Layers=2,Concatenated", pw.pairwise_square_bert, {"model":bert_model_base, "tokenizer":bert_tokenizer_base, "ids_to_texts":descriptions, "metric":"cosine", "method":"concat", "layers":2}, spatial.distance.cosine),
    #Method("BERT", " Base:Layers=3,Concatenated", pw.pairwise_square_bert, {"model":bert_model_base, "tokenizer":bert_tokenizer_base, "ids_to_texts":descriptions, "metric":"cosine", "method":"concat", "layers":3}, spatial.distance.cosine),
    #Method("BERT", " Base:Layers=4,Concatenated", pw.pairwise_square_bert, {"model":bert_model_base, "tokenizer":bert_tokenizer_base, "ids_to_texts":descriptions, "metric":"cosine", "method":"concat", "layers":4}, spatial.distance.cosine),
    #Method("BERT", " Base:Layers=2,Summed", pw.pairwise_square_bert, {"model":bert_model_base, "tokenizer":bert_tokenizer_base, "ids_to_texts":descriptions, "metric":"cosine", "method":"sum", "layers":2}, spatial.distance.cosine),
    #Method("BERT", " Base:Layers=3,Summed", pw.pairwise_square_bert, {"model":bert_model_base, "tokenizer":bert_tokenizer_base, "ids_to_texts":descriptions, "metric":"cosine", "method":"sum", "layers":3}, spatial.distance.cosine),
    #Method("BERT", " Base:Layers=4,Summed", pw.pairwise_square_bert, {"model":bert_model_base, "tokenizer":bert_tokenizer_base, "ids_to_texts":descriptions, "metric":"cosine", "method":"sum", "layers":4}, spatial.distance.cosine),
    #Method("BioBERT", "PMC,Layers=2,Concatenated", pw.pairwise_square_bert, {"model":bert_model_pmc, "tokenizer":bert_tokenizer_pmc, "ids_to_texts":descriptions, "metric":"cosine", "method":"concat", "layers":2}, spatial.distance.cosine),
    #Method("BioBERT", "PMC,Layers=3,Concatenated", pw.pairwise_square_bert, {"model":bert_model_pmc, "tokenizer":bert_tokenizer_pmc, "ids_to_texts":descriptions, "metric":"cosine", "method":"concat", "layers":3}, spatial.distance.cosine),
    #Method("BioBERT", "PMC,Layers=4,Concatenated", pw.pairwise_square_bert, {"model":bert_model_pmc, "tokenizer":bert_tokenizer_pmc, "ids_to_texts":descriptions, "metric":"cosine", "method":"concat", "layers":4}, spatial.distance.cosine),
    #Method("BioBERT", "PubMed,Layers=4,Concatenated", pw.pairwise_square_bert, {"model":bert_model_pubmed, "tokenizer":bert_tokenizer_pubmed, "ids_to_texts":descriptions, "metric":"cosine", "method":"concat", "layers":4}, spatial.distance.cosine),
    #Method("BioBERT", "PubMed,PMC,Layers=4,Concatenated", pw.pairwise_square_bert, {"model":bert_model_pubmed_pmc, "tokenizer":bert_tokenizer_pubmed_pmc, "ids_to_texts":descriptions, "metric":"cosine", "method":"concat", "layers":4}, spatial.distance.cosine),

    # Methods that use variations on the n-grams approach with full preprocessing (includes stemming).
    Method("N-Grams", "Full,Words,1-grams,2-grams", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_full_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,2),"max_features":10000, "tfidf":False}, spatial.distance.cosine),
    Method("N-Grams", "Full,Words,1-grams,2-grams,Binary", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_full_preprocessing, "metric":"jaccard", "binary":True, "analyzer":"word", "ngram_range":(1,2), "max_features":10000, "tfidf":False}, spatial.distance.jaccard),
    Method("N-Grams", "Full,Words,1-grams", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_full_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.cosine),
    Method("N-Grams", "Full,Words,1-grams,Binary", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_full_preprocessing, "metric":"jaccard", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.jaccard),
    Method("N-Grams", "Full,Words,1-grams,2-grams,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_full_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,2),"max_features":10000, "tfidf":True}, spatial.distance.cosine),
    Method("N-Grams", "Full,Words,1-grams,2-grams,Binary,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_full_preprocessing, "metric":"cosine", "binary":True, "analyzer":"word", "ngram_range":(1,2), "max_features":10000, "tfidf":True}, spatial.distance.cosine),
    Method("N-Grams", "Full,Words,1-grams,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_full_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine),
    Method("N-Grams", "Full,Words,1-grams,Binary,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_full_preprocessing, "metric":"cosine", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine),

    # Methods that use variations on the n-grams approach with simple preprocessing (no stemming).
    Method("N-Grams", "Simple,Words,1-grams,2-grams", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_simple_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,2),"max_features":10000, "tfidf":False}, spatial.distance.cosine),
    Method("N-Grams", "Simple,Words,1-grams,2-grams,Binary", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_simple_preprocessing, "metric":"jaccard", "binary":True, "analyzer":"word", "ngram_range":(1,2), "max_features":10000, "tfidf":False}, spatial.distance.jaccard),
    Method("N-Grams", "Simple,Words,1-grams", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_simple_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.cosine),
    Method("N-Grams", "Simple,Words,1-grams,Binary", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_simple_preprocessing, "metric":"jaccard", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.jaccard),
    Method("N-Grams", "Simple,Words,1-grams,2-grams,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_simple_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,2),"max_features":10000, "tfidf":True}, spatial.distance.cosine),
    Method("N-Grams", "Simple,Words,1-grams,2-grams,Binary,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_simple_preprocessing, "metric":"cosine", "binary":True, "analyzer":"word", "ngram_range":(1,2), "max_features":10000, "tfidf":True}, spatial.distance.cosine),
    Method("N-Grams", "Simple,Words,1-grams,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_simple_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine),
    Method("N-Grams", "Simple,Words,1-grams,Binary,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_simple_preprocessing, "metric":"cosine", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine),

    # Methods that use variations on the n-grams approach selecting for specific parts-of-speech (includes stemming).
    Method("N-Grams", "Full,Nouns,1-grams", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_noun_only_full_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.cosine),
    Method("N-Grams", "Full,Nouns,1-grams,Binary", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_noun_only_full_preprocessing, "metric":"jaccard", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.jaccard),
    Method("N-Grams", "Full,Nouns,1-grams,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_noun_only_full_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine),
    Method("N-Grams", "Full,Nouns,1-grams,Binary,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_noun_only_full_preprocessing, "metric":"cosine", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine),
    Method("N-Grams", "Full,Adjectives,1-grams", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_adj_only_full_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.cosine),
    Method("N-Grams", "Full,Adjectives,1-grams,Binary", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_adj_only_full_preprocessing, "metric":"jaccard", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.jaccard),
    Method("N-Grams", "Full,Adjectives,1-grams,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_adj_only_full_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine),
    Method("N-Grams", "Full,Adjectives,1-grams,Binary,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_adj_only_full_preprocessing, "metric":"cosine", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine),
    Method("N-Grams", "Full,Adjectives,1-grams", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_noun_adj_full_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.cosine),
    Method("N-Grams", "Full,Adjectives,1-grams,Binary", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_noun_adj_full_preprocessing, "metric":"jaccard", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.jaccard),
    Method("N-Grams", "Full,Adjectives,1-grams,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_noun_adj_full_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine),
    Method("N-Grams", "Full,Adjectives,1-grams,Binary,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_noun_adj_full_preprocessing, "metric":"cosine", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine),

    # Methods that use variations on the n-grams approach with a reduced vocabulary size and simple preprocessing (no stemming).
    Method("N-Grams", "Full,Words,Linares Pontes,1-grams", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_linares_pontes, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.cosine),
    Method("N-Grams", "Full,Words,Linares Pontes,1-grams,Binary", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_linares_pontes, "metric":"jaccard", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.jaccard),
    Method("N-Grams", "Full,Words,Linares Pontes,1-grams,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_linares_pontes, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine),
    Method("N-Grams", "Full,Words,Linares Pontes,1-grams,Binary,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":descriptions_linares_pontes, "metric":"cosine", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine),



    # Methods that use terms inferred from automated annotation of the text.
    Method("NOBLE Coder", "Precise", pw.pairwise_square_annotations, {"ids_to_annotations":annotations_noblecoder_precise, "ontology":ontology, "binary":True, "metric":"jaccard", "tfidf":False}, spatial.distance.jaccard),
    Method("NOBLE Coder", "Partial", pw.pairwise_square_annotations, {"ids_to_annotations":annotations_noblecoder_partial, "ontology":ontology, "binary":True, "metric":"jaccard", "tfidf":False}, spatial.distance.jaccard),
    Method("NOBLE Coder", "Precise,TFIDF", pw.pairwise_square_annotations, {"ids_to_annotations":annotations_noblecoder_precise, "ontology":ontology, "binary":True, "metric":"cosine", "tfidf":True}, spatial.distance.cosine),
    Method("NOBLE Coder", "Partial,TFIDF", pw.pairwise_square_annotations, {"ids_to_annotations":annotations_noblecoder_partial, "ontology":ontology, "binary":True, "metric":"cosine", "tfidf":True}, spatial.distance.cosine),

    # Methods that use terms assigned by humans that are present in the dataset.
    Method("GO", "Default", pw.pairwise_square_annotations, {"ids_to_annotations":go_annotations, "ontology":ontology, "metric":"jaccard", "binary":True, "analyzer":"word", "ngram_range":(1,1), "tfidf":False}, spatial.distance.jaccard),
    Method("PO", "Default", pw.pairwise_square_annotations, {"ids_to_annotations":po_annotations, "ontology":ontology, "metric":"jaccard", "binary":True, "analyzer":"word", "ngram_range":(1,1), "tfidf":False}, spatial.distance.jaccard),






    # Individual phenes from those larger phenotypes.

    # Approaches were the phenotype descriptions were split into individual phenes first (computationally expensive).
    Method("Doc2Vec (Phenes)", "Wikipedia,Size=300", pw.pairwise_square_doc2vec, {"model":doc2vec_wiki_model, "ids_to_texts":phenes, "metric":"cosine"}, spatial.distance.cosine, tag="phenes"),
    Method("Doc2Vec (Phenes)", "PubMed,Size=100", pw.pairwise_square_doc2vec, {"model":doc2vec_pubmed_model, "ids_to_texts":phenes, "metric":"cosine"}, spatial.distance.cosine, tag="phenes"),
    Method("Word2Vec (Phenes)", "Wikipedia,Size=300,Mean", pw.pairwise_square_word2vec, {"model":word2vec_model, "ids_to_texts":phenes, "metric":"cosine", "method":"mean"}, spatial.distance.cosine, tag="phenes"),
    Method("Word2Vec (Phenes)", "Wikipedia,Size=300,Max", pw.pairwise_square_word2vec, {"model":word2vec_model, "ids_to_texts":phenes, "metric":"cosine", "method":"max"}, spatial.distance.cosine, tag="phenes"),

    #Method("BERT (Phenes)", "Base:Layers=2,Concatenated", pw.pairwise_square_bert, {"model":bert_model_base, "tokenizer":bert_tokenizer_base, "ids_to_texts":phenes, "metric":"cosine", "method":"concat", "layers":2}, spatial.distance.cosine, tag="phenes"),
    #Method("BERT (Phenes)", "Base:Layers=3,Concatenated", pw.pairwise_square_bert, {"model":bert_model_base, "tokenizer":bert_tokenizer_base, "ids_to_texts":phenes, "metric":"cosine", "method":"concat", "layers":3}, spatial.distance.cosine, tag="phenes"),
    #Method("BERT (Phenes)", "Base:Layers=4,Concatenated", pw.pairwise_square_bert, {"model":bert_model_base, "tokenizer":bert_tokenizer_base, "ids_to_texts":phenes, "metric":"cosine", "method":"concat", "layers":4}, spatial.distance.cosine, tag="phenes"),
    #Method("BERT (Phenes)", "Base:Layers=2,Summed", pw.pairwise_square_bert, {"model":bert_model_base, "tokenizer":bert_tokenizer_base, "ids_to_texts":phenes, "metric":"cosine", "method":"sum", "layers":2}, spatial.distance.cosine, tag="phenes"),
    #Method("BERT (Phenes)", "Base:Layers=3,Summed", pw.pairwise_square_bert, {"model":bert_model_base, "tokenizer":bert_tokenizer_base, "ids_to_texts":phenes, "metric":"cosine", "method":"sum", "layers":3}, spatial.distance.cosine, tag="phenes"),
    #Method("BERT (Phenes)", "Base:Layers=4,Summed", pw.pairwise_square_bert, {"model":bert_model_base, "tokenizer":bert_tokenizer_base, "ids_to_texts":phenes, "metric":"cosine", "method":"sum", "layers":4}, spatial.distance.cosine, tag="phenes"),
    #Method("BioBERT (Phenes)", "PMC,Layers=2,Concatenated", pw.pairwise_square_bert, {"model":bert_model_pmc, "tokenizer":bert_tokenizer_pmc, "ids_to_texts":phenes, "metric":"cosine", "method":"concat", "layers":2}, spatial.distance.cosine, tag="phenes"),
    #Method("BioBERT (Phenes)", "PMC,Layers=3,Concatenated", pw.pairwise_square_bert, {"model":bert_model_pmc, "tokenizer":bert_tokenizer_pmc, "ids_to_texts":phenes, "metric":"cosine", "method":"concat", "layers":3}, spatial.distance.cosine, tag="phenes"),
    #Method("BioBERT (Phenes)", "PMC,Layers=4,Concatenated", pw.pairwise_square_bert, {"model":bert_model_pmc, "tokenizer":bert_tokenizer_pmc, "ids_to_texts":phenes, "metric":"cosine", "method":"concat", "layers":4}, spatial.distance.cosine, tag="phenes"),
    #Method("BioBERT (Phenes)", "PubMed,Layers=4,Concatenated", pw.pairwise_square_bert, {"model":bert_model_pubmed, "tokenizer":bert_tokenizer_pubmed, "ids_to_texts":phenes, "metric":"cosine", "method":"concat", "layers":4}, spatial.distance.cosine, tag="phenes"),
    #Method("BioBERT (Phenes)", "PubMed,PMC,Layers=4,Concatenated", pw.pairwise_square_bert, {"model":bert_model_pubmed_pmc, "tokenizer":bert_tokenizer_pubmed_pmc, "ids_to_texts":phenes, "metric":"cosine", "method":"concat", "layers":4}, spatial.distance.cosine, tag="phenes"),



    # Methods that use variations on the N-Grams (Phenes) approach with full preprocessing (includes stemming).
    #Method("N-Grams (Phenes)", "Full,Words,1-grams,2-grams", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_full_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,2),"max_features":10000, "tfidf":False}, spatial.distance.cosine, tag="phenes"),
    #Method("N-Grams (Phenes)", "Full,Words,1-grams,2-grams,Binary", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_full_preprocessing, "metric":"jaccard", "binary":True, "analyzer":"word", "ngram_range":(1,2), "max_features":10000, "tfidf":False}, spatial.distance.jaccard, tag="phenes"),
    #Method("N-Grams (Phenes)", "Full,Words,1-grams", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_full_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.cosine, tag="phenes"),
    Method("N-Grams (Phenes)", "Full,Words,1-grams,Binary", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_full_preprocessing, "metric":"jaccard", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.jaccard, tag="phenes"),
    #Method("N-Grams (Phenes)", "Full,Words,1-grams,2-grams,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_full_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,2),"max_features":10000, "tfidf":True}, spatial.distance.cosine, tag="phenes"),
    #Method("N-Grams (Phenes)", "Full,Words,1-grams,2-grams,Binary,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_full_preprocessing, "metric":"cosine", "binary":True, "analyzer":"word", "ngram_range":(1,2), "max_features":10000, "tfidf":True}, spatial.distance.cosine, tag="phenes"),
    #Method("N-Grams (Phenes)", "Full,Words,1-grams,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_full_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine, tag="phenes"),
    Method("N-Grams (Phenes)", "Full,Words,1-grams,Binary,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_full_preprocessing, "metric":"cosine", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine, tag="phenes"),

    # Methods that use variations on the N-Grams (Phenes) approach with simple preprocessing (no stemming).
    #Method("N-Grams (Phenes)", "Simple,Words,1-grams,2-grams", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_simple_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,2),"max_features":10000, "tfidf":False}, spatial.distance.cosine, tag="phenes"),
    #Method("N-Grams (Phenes)", "Simple,Words,1-grams,2-grams,Binary", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_simple_preprocessing, "metric":"jaccard", "binary":True, "analyzer":"word", "ngram_range":(1,2), "max_features":10000, "tfidf":False}, spatial.distance.jaccard, tag="phenes"),
    #Method("N-Grams (Phenes)", "Simple,Words,1-grams", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_simple_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.cosine, tag="phenes"),
    Method("N-Grams (Phenes)", "Simple,Words,1-grams,Binary", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_simple_preprocessing, "metric":"jaccard", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.jaccard, tag="phenes"),
    #Method("N-Grams (Phenes)", "Simple,Words,1-grams,2-grams,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_simple_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,2),"max_features":10000, "tfidf":True}, spatial.distance.cosine, tag="phenes"),
    #Method("N-Grams (Phenes)", "Simple,Words,1-grams,2-grams,Binary,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_simple_preprocessing, "metric":"cosine", "binary":True, "analyzer":"word", "ngram_range":(1,2), "max_features":10000, "tfidf":True}, spatial.distance.cosine, tag="phenes"),
    #Method("N-Grams (Phenes)", "Simple,Words,1-grams,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_simple_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine, tag="phenes"),
    Method("N-Grams (Phenes)", "Simple,Words,1-grams,Binary,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_simple_preprocessing, "metric":"cosine", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine, tag="phenes"),

    # Methods that use variations on the N-Grams (Phenes) approach selecting for specific parts-of-speech (includes stemming).
    #Method("N-Grams (Phenes)", "Full,Nouns,1-grams", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_noun_only_full_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.cosine, tag="phenes"),
    #Method("N-Grams (Phenes)", "Full,Nouns,1-grams,Binary", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_noun_only_full_preprocessing, "metric":"jaccard", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.jaccard, tag="phenes"),
    #Method("N-Grams (Phenes)", "Full,Nouns,1-grams,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_noun_only_full_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine, tag="phenes"),
    #Method("N-Grams (Phenes)", "Full,Nouns,1-grams,Binary,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_noun_only_full_preprocessing, "metric":"cosine", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine, tag="phenes"),
    #Method("N-Grams (Phenes)", "Full,Adjectives,1-grams", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_adj_only_full_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.cosine, tag="phenes"),
    #Method("N-Grams (Phenes)", "Full,Adjectives,1-grams,Binary", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_adj_only_full_preprocessing, "metric":"jaccard", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.jaccard, tag="phenes"),
    #Method("N-Grams (Phenes)", "Full,Adjectives,1-grams,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_adj_only_full_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine, tag="phenes"),
    #Method("N-Grams (Phenes)", "Full,Adjectives,1-grams,Binary,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_adj_only_full_preprocessing, "metric":"cosine", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine, tag="phenes"),
    #Method("N-Grams (Phenes)", "Full,Adjectives,1-grams", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_noun_adj_full_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.cosine, tag="phenes"),
    #Method("N-Grams (Phenes)", "Full,Adjectives,1-grams,Binary", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_noun_adj_full_preprocessing, "metric":"jaccard", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.jaccard, tag="phenes"),
    #Method("N-Grams (Phenes)", "Full,Adjectives,1-grams,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_noun_adj_full_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine, tag="phenes"),
    #Method("N-Grams (Phenes)", "Full,Adjectives,1-grams,Binary,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_noun_adj_full_preprocessing, "metric":"cosine", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine, tag="phenes"),

    # Methods that use variations on the N-Grams (Phenes) approach with a reduced vocabulary size and simple preprocessing (no stemming).
    #Method("N-Grams (Phenes)", "Full,Words,Linares Pontes,1-grams", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_linares_pontes, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.cosine, tag="phenes"),
    Method("N-Grams (Phenes)", "Full,Words,Linares Pontes,1-grams,Binary", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_linares_pontes, "metric":"jaccard", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":False}, spatial.distance.jaccard, tag="phenes"),
    #Method("N-Grams (Phenes)", "Full,Words,Linares Pontes,1-grams,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_linares_pontes, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine, tag="phenes"),
    #Method("N-Grams (Phenes)", "Full,Words,Linares Pontes,1-grams,Binary,TFIDF", pw.pairwise_square_ngrams, {"ids_to_texts":phenes_linares_pontes, "metric":"cosine", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":10000, "tfidf":True}, spatial.distance.cosine, tag="phenes"),

]


# <a id="running"></a>
# ### Running all of the methods to generate distance matrices
# Something here if needed.

# In[18]:


# Generate all the pairwise distance matrices (not in parallel).
graphs = {}
names = []
durations = []
for method in methods:
    graph,duration = function_wrapper_with_duration(function=method.function, args=method.kwargs)
    graphs[method.name_with_hyperparameters] = graph
    names.append(method.name_with_hyperparameters)
    durations.append(to_hms(duration))
    print("{:60} {}".format(method.name_with_hyperparameters,to_hms(duration)))
durations_df = pd.DataFrame({"method":names,"duration":durations})
durations_df.to_csv(os.path.join(OUTPUT_DIR,"part_4_durations.csv"), index=False)


# <a id="merging"></a>
# ### Merging all of the distance matrices into a single dataframe specifying edges
# This section also handles replacing IDs from the individual methods that are references individual phenes that are part of a larger phenotype, and replacing those IDs with IDs referencing the full phenotypes (one-to-one relationship between phenotypes and genes). In this case, the minimum distance found between any two phenes from those two phenotypes represents the distance between that pair of phenotypes.

# In[19]:


# Merging all the edgelists together.
metric_dict = {method.name_with_hyperparameters:method.metric for method in methods}
tags_dict = {method.name_with_hyperparameters:method.tag for method in methods}
names = list(graphs.keys())
edgelists = {k:v.edgelist for k,v in graphs.items()}

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


# <a id="ensemble"></a>
# ### Combining multiple distances measurements into summarizing distance values
# The purpose of this section is to iteratively train models on subsections of the dataset using simple regression or machine learning approaches to predict a value from zero to one indicating indicating how likely is it that two genes share atleast one of the specified groups in common. The information input to these models is the distance scores provided by each method in some set of all the methods used in this notebook. The purpose is to see whether or not a function of these similarity scores specifically trained to the task of predicting common groupings is better able to used the distance metric information to report a score for this task.

# In[20]:


# Get the average distance percentile as a means of combining multiple scores.
name = "Mean"
df[name] = df[names].rank(pct=True).mean(axis=1)
names.append(name)
df.head(20)


# In[21]:


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


# <a id="cluster_analysis"></a>
# # Part 5. Cluster Analysis
# The purpose of this section is to look at different ways that the embeddings obtained for the dataset of phenotype descriptions can be used to cluster or organize the genes to which those phenotypes are mapped into subgroups or representations. These approaches include generating topic models from the data, and doing agglomerative clustering to find clusters to which each gene belongs.

# <a id="topic_modeling"></a>
# ### Approach 1: Topic modeling based on n-grams with a reduced vocabulary
# Topic modelling learns a set of word probability distributions from the dataset of text descriptions, which represent distinct topics which are present in the dataset. Each text description can then be represented as a discrete probability distribution over the learned topics based on the probability that a given piece of text belongs to each particular topics. This is a form of data reduction because a high dimensionsal bag-of-words can be represented as a vector of *k* probabilities where *k* is the number of topics. The main advantages of topic modelling over clustering is that topic modelling provides soft classifications that can be additionally interpreted, rather than hard classifications into a single cluster. Topic models are also explainable, because the word probability distributions for that topic can be used to determine which words are most representative of any given topic. One problem with topic modelling is that is uses the n-grams embeddings to semantic similarity between different words is not accounted for. To help alleviate this, this section uses implementations of some existing algorithms to compress the vocabulary as a preprocessing step based on word distance matrices generated using word embeddings.

# In[22]:


# Get a list of texts to create a topic model from, from one of the processed description dictionaries above. 
texts = [description for i,description in descriptions_linares_pontes.items()]

# Creating and fitting the topic model, either NFM or LDA.
number_of_topics = 42
seed = 0
vectorizer = TfidfVectorizer(max_features=10000, stop_words="english", max_df=0.95, min_df=2, lowercase=False)
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
    texts = [descriptions_linares_pontes[i] for i in ids]
    topic_vectors = get_topic_embeddings(texts, cls, vectorizer)
    mean_topic_vector = np.mean(topic_vectors, axis=0)
    group_to_topic_vector[group_id] = mean_topic_vector
    
tm_df = pd.DataFrame(group_to_topic_vector)

# Changing the order of the Lloyd, Meinke phenotype subsets to match other figures for consistency.
if NOTEBOOK_TAGS["subsets"]:
    filename = "../data/group_related_files/lloyd/lloyd_function_hierarchy_irb_cleaned.csv"
    lmtm_df = pd.read_csv(filename)    
    columns_in_order = [col for col in lmtm_df["Subset Symbol"].values if col in tm_df.columns]
    tm_df = tm_df[columns_in_order]
    
# Reordering so consistency with the curated subsets can be checked by looking at the diagonal.
tm_df["idxmax"] = tm_df.idxmax(axis = 1)
tm_df["idxmax"] = tm_df["idxmax"].apply(lambda x: tm_df.columns.get_loc(x))
tm_df = tm_df.sort_values(by="idxmax")
tm_df.drop(columns=["idxmax"], inplace=True)
tm_df = tm_df.reset_index(drop=False).rename({"index":"topic"},axis=1).reset_index(drop=False).rename({"index":"order"},axis=1)
tm_df.to_csv(os.path.join(OUTPUT_DIR,"part_5_topic_modeling.csv"), index=False)
tm_df


# In[23]:


# Describing what the most representative tokens for each topic in the model are.
num_top_words = 2
feature_names = vectorizer.get_feature_names()
for i,topic_vec in enumerate(cls.components_):
    print(i,end=": ")
    for fid in topic_vec.argsort()[-1:-num_top_words-1:-1]:
        word = feature_names[fid]
        word = " ".join(unreduce_lp[word])
        print(word, end=" ")
    print()


# <a id="clustering"></a>
# ### Approach 2: Agglomerative clustering and comparison to predefined groups
# This clustering approach uses agglomerative clustering to cluster the genes into a fixed number of clusters based off the distances between their embedding representations using all of the above methods. Clustering into a fixed number of clusters allows for clustering into a similar number of groups as a present in some existing grouping of the data, such as phenotype categories or biochemical pathways, and then determining if the clusters obtained are at all similar to the groupings that already exist.

# In[24]:


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
if NOTEBOOK_TAGS["subsets"]:
    filename = "../data/group_related_files/lloyd/lloyd_function_hierarchy_irb_cleaned.csv"
    lmtm_df = pd.read_csv(filename)    
    columns_in_order = [col for col in lmtm_df["Subset Symbol"].values if col in tm_df.columns]
    tm_df = tm_df[columns_in_order]

# Reordering so consistency with the curated subsets can be checked by looking at the diagonal.
ac_df["idxmax"] = ac_df.idxmax(axis = 1)
ac_df["idxmax"] = ac_df["idxmax"].apply(lambda x: ac_df.columns.get_loc(x))
ac_df = ac_df.sort_values(by="idxmax")
ac_df.drop(columns=["idxmax"], inplace=True)
ac_df = ac_df.reset_index(drop=False).rename({"index":"cluster"},axis=1).reset_index(drop=False).rename({"index":"order"},axis=1)
ac_df.to_csv(os.path.join(OUTPUT_DIR,"part_5_agglomerative_clustering.csv"), index=False)
ac_df


# <a id="phenologs"></a>
# ### Approach 3: Looking for phenolog relationships between clusters and OMIM disease phenotypes
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
print(omim_df.groupby("compressed_phenotype_mim_name").size())


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
#phenolog_df.sort_values(by=["x"], inplace=True, ascending=False)
phenolog_df.sort_values(by=["p_value"], inplace=True, ascending=True)
phenolog_df = phenolog_df[["omim_phenotype_name", "cluster", "size", "x", "p_value", "p_adjusted", "candidate_gene_symbols"]]
phenolog_df.to_csv(os.path.join(OUTPUT_DIR,"part_5_phenologs.csv"), index=False)
phenolog_df.head(30)


# ### Approach 4: Agglomerative clustering and sillhouette scores for each NLP method

# In[ ]:


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
        #to_id = array_index_to_id
        ac = AgglomerativeClustering(n_clusters=n, linkage="complete", affinity="precomputed")
        clustering = ac.fit(distance_matrix)
        sil_score = silhouette_score(distance_matrix, clustering.labels_, metric="precomputed")
        n_clusters_silhouette_scores[name][n] = sil_score
sil_df = pd.DataFrame(n_clusters_silhouette_scores).reset_index(drop=False).rename({"index":"n"},axis="columns")
sil_df.to_csv(os.path.join(OUTPUT_DIR,"part_5_silhouette_scores_by_n.csv"), index=False)
sil_df.head(10)


# # Part 6. Supervised Tasks

# <a id="merging"></a>
# ### Option 1: Merging in the previously curated similarity values from the Oellrich, Walls et al. (2015) dataset
# This section reads in a file that contains the previously calculated distance values from the Oellrich, Walls et al. (2015) dataset, and merges it with the values which are obtained here for all of the applicable natural language processing or machine learning methods used, so that the graphs which are specified by these sets of distances values can be evaluated side by side in the subsequent sections.

# In[ ]:


# Add a column that indicates the distance estimated using curated EQ statements.
df = df.merge(right=pppn_edgelist.df, how="left", on=["from","to"])
df.fillna(value=0.000,inplace=True)
df.rename(columns={"value":"EQs"}, inplace=True)
df["EQs"] = 1-df["EQs"]
names.append("EQs")
df.head(10)


# ### Option 2: Merging with information about shared biochemical pathways or groups.
# The relevant information for each edge includes questions like whether or not the two genes that edge connects share a group or biochemical pathway in common, or if those genes are from the same species. This information can then later be used as the target values for predictive models, or for filtering the graphs represented by these edge lists. Either the grouping information or the protein-protein interaction information should be used.

# In[ ]:


# Column indicating whether or not the two genes share this features (e.g., pathway in common, same group).
df["shared"] = df[["from","to"]].apply(lambda x: len(set(id_to_group_ids[x["from"]]).intersection(set(id_to_group_ids[x["to"]])))>0, axis=1)*1
# Column indicating whether the two genes are from the same species.
species_dict = dataset.get_species_dictionary()
df["same"] = df[["from","to"]].apply(lambda x: species_dict[x["from"]]==species_dict[x["to"]],axis=1)*1
print(Counter(df["shared"].values))
print(Counter(df["same"].values))


# ### Option 3: Merging with information about protein-protein interactions.

# # Merging information from the protein-protein interaction database with this dataset.
# df = df.merge(right=string_data.df, how="left", on=["from","to"])
# df.fillna(value=0,inplace=True)
# df["shared"] = (df["combined_score"] != 0.00)*1
# df.tail(12)

# <a id="ks"></a>
# ### Do the edges joining genes that share a group, pathway, or interaction come from a different distribution?
# The purpose of this section is to visualize kernel estimates for the distributions of distance or similarity scores generated by each of the methods tested for measuring semantic similarity or generating vector representations of the phenotype descriptions. Ideally, better methods should show better separation betwene the distributions for distance values between two genes involved in a common specified group or two genes that are not. Additionally, a statistical test is used to check whether these two distributions are significantly different from each other or not, although this is a less informative measure than the other tests used in subsequent sections, because it does not address how useful these differences in the distributions actually are for making predictions about group membership.

# In[ ]:


# Use Kolmogorov-Smirnov test to see if edges between genes that share a group come from a distinct distribution.
ppi_pos_dict = {name:(df[df["shared"] > 0.00][name].values) for name in names}
ppi_neg_dict = {name:(df[df["shared"] == 0.00][name].values) for name in names}
for name in names:
    stat,p = ks_2samp(ppi_pos_dict[name],ppi_neg_dict[name])
    pos_mean = np.average(ppi_pos_dict[name])
    neg_mean = np.average(ppi_neg_dict[name])
    pos_n = len(ppi_pos_dict[name])
    neg_n = len(ppi_neg_dict[name])
    TABLE[name].update({"mean_1":pos_mean, "mean_0":neg_mean, "n_1":pos_n, "n_0":neg_n})
    TABLE[name].update({"ks":stat, "ks_pval":p})
    
    
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


# <a id="within"></a>
# ### Looking at within-group or within-pathway distances in each graph
# The purpose of this section is to determine which methods generated graphs which tightly group genes which share common pathways or group membership with one another. In order to compare across different methods where the distance value distributions are different, the mean distance values for each group for each method are convereted to percentile scores. Lower percentile scores indicate that the average distance value between any two genes that belong to that group is lower than most of the distance values in the entire distribution for that method.

# In[ ]:


# Get all the average within-pathway phenotype distance values for each method for each particular pathway.
group_id_to_ids = groups.get_group_id_to_ids_dict(dataset.get_gene_dictionary())
group_ids = list(group_id_to_ids.keys())
graph = IndexedGraph(df)
within_weights_dict = defaultdict(lambda: defaultdict(list))
within_percentiles_dict = defaultdict(lambda: defaultdict(list))
all_weights_dict = {}
for name in names:
    all_weights_dict[name] = df[name].values
    for group in group_ids:
        within_ids = group_id_to_ids[group]
        within_pairs = [(i,j) for i,j in itertools.permutations(within_ids,2)]
        mean_weight = np.mean((graph.get_values(within_pairs, kind=name)))
        within_weights_dict[name][group] = mean_weight
        within_percentiles_dict[name][group] = stats.percentileofscore(df[name].values, mean_weight, kind="rank")

# Generating a dataframe of percentiles of the mean in-group distance scores.
within_dist_data = pd.DataFrame(within_percentiles_dict)
within_dist_data = within_dist_data.dropna(axis=0, inplace=False)
within_dist_data = within_dist_data.round(4)

# Adding relevant information to this dataframe and saving.
within_dist_data["mean_rank"] = within_dist_data.rank().mean(axis=1)
within_dist_data["mean_percentile"] = within_dist_data.mean(axis=1)
within_dist_data.sort_values(by="mean_percentile", inplace=True)
within_dist_data.reset_index(inplace=True)
within_dist_data["group_id"] = within_dist_data["index"]
within_dist_data["full_name"] = within_dist_data["group_id"].apply(lambda x: groups.get_long_name(x))
within_dist_data["n"] = within_dist_data["group_id"].apply(lambda x: len(group_id_to_ids[x]))
within_dist_data = within_dist_data[flatten(["group_id","full_name","n","mean_percentile","mean_rank",names])]
within_dist_data.to_csv(os.path.join(OUTPUT_DIR,"part_6_within_distances.csv"), index=False)
within_dist_data.head(5)


# <a id="auc"></a>
# ### Predicting whether two genes belong to the same group, pathway, or share an interaction
# The purpose of this section is to see if whether or not two genes share atleast one common pathway can be predicted from the distance scores assigned using analysis of text similarity. The evaluation of predictability is done by reporting a precision and recall curve for each method, as well as remembering the area under the curve, and ratio between the area under the curve and the baseline (expected area when guessing randomly) for each method.

# In[ ]:


y_true_dict = {name:df["shared"] for name in names}
y_prob_dict = {name:(1 - df[name].values) for name in names}
num_plots, plots_per_row, row_width, row_height = (len(names), 4, 14, 3)
fig,axs = plt.subplots(math.ceil(num_plots/plots_per_row), plots_per_row, squeeze=False)
for name,ax in zip(names, axs.flatten()):
    
    # Obtaining the values and metrics.
    y_true, y_prob = y_true_dict[name], y_prob_dict[name]
    n_pos, n_neg = Counter(y_true)[1], Counter(y_true)[0]
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    baseline = Counter(y_true)[1]/len(y_true) 
    area = auc(recall, precision)
    auc_to_baseline_auc_ratio = area/baseline
    TABLE[name].update({"auc":area, "baseline":baseline, "ratio":auc_to_baseline_auc_ratio})

    # Producing the precision recall curve.
    step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
    ax.step(recall, precision, color='black', alpha=0.2, where='post')
    ax.fill_between(recall, precision, alpha=0.7, color='black', **step_kwargs)
    ax.axhline(baseline, linestyle="--", color="lightgray")
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.set_title("PR {0} (Baseline={1:0.3f})".format(name, baseline))
    
fig.set_size_inches(row_width, row_height*math.ceil(num_plots/plots_per_row))
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR,"part_6_prcurve_shared.png"),dpi=400)
plt.close()


# <a id="y"></a>
# ### Are genes in the same group or pathway ranked higher with respect to individual nodes?
# This is a way of statistically seeing if for some value k, the graph ranks more edges from some particular gene to any other gene that it has a true protein-protein interaction with higher or equal to rank k, than we would expect due to random chance. This way of looking at the problem helps to be less ambiguous than the previous methods, because it gets at the core of how this would actually be used. In other words, we don't really care how much true information we're missing as long as we're still able to pick up some new useful information by building these networks, so even though we could be missing a lot, what's going on at the very top of the results? These results should be comparable to very strictly thresholding the network and saying that the remaining edges are our guesses at interactions. This is comparable to just looking at the far left-hand side of the precision recall curves, but just quantifies it slightly differently.

# In[ ]:


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

# In[ ]:


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


# In[ ]:


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

# In[ ]:


results = pd.DataFrame(TABLE).transpose()
columns = flatten(["Hyperparams","Group","Order","Topic","Data",results.columns])
results["Hyperparams"] = ""
results["Group"] = ""
results["Order"] = np.arange(results.shape[0])
results["Topic"] = TOPIC
results["Data"] = DATA
results = results[columns]
results.reset_index(inplace=True)
results = results.rename({"index":"Method"}, axis="columns")
hyperparam_sep = ":"
results["Hyperparams"] = results["Method"].map(lambda x: x.split(hyperparam_sep)[1] if hyperparam_sep in x else "None")
results["Method"] = results["Method"].map(lambda x: x.split(hyperparam_sep)[0])
results.to_csv(os.path.join(OUTPUT_DIR,"part_6_full_table.csv"), index=False)
results

