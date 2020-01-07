#!/usr/bin/env python
# coding: utf-8

# ## Table of Contents
# 
# - [Introduction](#part1)
# 
# - [Loading Data](#2)
#     - [Dataset of text descriptions](#3) (Gene names and identifiers associated with phenotype descriptions and ontology term annotations)
#     - [Dataset of groups or categories](#4) (Other sources of data, such as STRING and KEGG)
#     - [Relating the datasets](#5) (Creating mappings between genes and their mentions in external datasets)
#     - [Filtering the datasets](#6)
#     
# - [Loading Models](#part7)
#     - [Why use embedding models?]()
#     - [BERT and BioBERT]()
#     - [Word2Vec and Doc2Vec]()
#     
# - [NLP Choices](#part8)
#     - [Creating vocabularies](#part9) (Learning relevant words from ontology descriptions or text corpora)
#     - [POS Tagging]() (Filtering words in descriptions to only include nouns or adjectives)
#     - [Preprocessing descriptions]() (Choosing options specific to each type of vectorization or embedding method)
#     - [Annotating descriptions]() (Running annotation tools and drawing from high-confidence curated annotations)
#     
# - [Building a Distance Matrix]()
#     - [Defining a list of methods to use]()
#     - [Running each method]()
#     - [Adding additional information]()
#     - [Something else]()
# - [Analysis]()
#     - [Looking at differences in the distributions for each method]()
#     - [Looking at the AUC for each method for predicting class]()
#     - [Looking at some other test statistics based on querying]()
#     - [Putting together the full output table of metrics]()
#     

# ### Text mining analysis and network analysis of the Oellrich, Walls et al., (2015) dataset
# The purpose of this notebook is to evaluate the phenotype similarity network based off of hand-curated EQ statement annotations with GO, PO, PATO, and ChEBI terms with regards to how well this network clusters phenotypes related to genes which have known protein-protein interactions, have biochemical pathways in common, or are involved in common phenotype groupings. These properties of this network are compared against these properties in networks constructed from the same data but using natural language processing and machine learning approaches to evaluating text similarity, as well as combinations of these approaches using higher level models. Output from this notebook is saved with naming conventions according to when the notebook was run, and all outputs are saved as images or tables in CSV format.
# 
# The purpose of this notebook is to answer the question of how networks genereated using phenotypic-text similarity based approaches through either embedding, vocabulary presence, or ontology annotation compare to or relate to networks that specify known protein-protein interactions. The hypothesis that these networks are potentially related is based on the idea that if two proteins interact, they are likely to be acting in a common pathway with a common biological function. If the phenotypic outcome of this pathway is observable and documented, then similarites between text describing the mutant phenotype for these genes may coincide with direct protein-protein interactions. The different sections in this notebook correspond to different ways of determining if the graphs based on similarity between text descriptions, encodings of text descriptions, or annotations derived from text descriptions at all correspond to known protein-protein interactions in this dataset. The knowledge source about the protein-protein interactions for genes in this dataset is the STRING database. The available entries in the whole dataset are subset to include only the genes that correspond to proteins that are atleast mentioned in the STRING database. This ways if a protein-protein interaction is not specified between two of the remaining genes, it is not because no interactions at all are documented either of those genes. The following cells focus on setting up a dataframe which specifies edge lists specific to each similarity method, and also a protein-protein interaction score for the genes which correspond to those two given nodes in the graphs.

# In[6]:


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
from scipy.stats import ks_2samp
from sklearn.metrics import precision_recall_curve, f1_score, auc
from sklearn.model_selection import train_test_split, KFold
from scipy import spatial, stats
from nltk.corpus import brown
from nltk.tokenize import word_tokenize
from sklearn.neighbors import KNeighborsClassifier
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from gensim.parsing.preprocessing import strip_non_alphanum, stem_text
from gensim.parsing.preprocessing import preprocess_string
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import remove_stopwords

sys.path.append("../../oats")
from oats.utils.utils import save_to_pickle, load_from_pickle, merge_list_dicts, flatten, to_hms
from oats.datasets.dataset import Dataset
from oats.datasets.groupings import Groupings
from oats.annotation.ontology import Ontology
from oats.datasets.string import String
from oats.datasets.edges import Edges
from oats.annotation.annotation import annotate_using_noble_coder
from oats.graphs import pairwise as pw
from oats.graphs.indexed import IndexedGraph
from oats.graphs.models import train_logistic_regression_model, apply_logistic_regression_model
from oats.graphs.models import train_random_forest_model, apply_random_forest_model
from oats.nlp.vocabulary import get_overrepresented_tokens, build_vocabulary_from_tokens
from oats.utils.utils import function_wrapper_with_duration
from oats.nlp.preprocess import concatenate_with_bar_delim

mpl.rcParams["figure.dpi"] = 400
warnings.simplefilter('ignore')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
nltk.download('punkt', quiet=True)
nltk.download('brown', quiet=True)


# <a id="part1"></a>
# ### Setting up the output table and input and output filepaths
# This section defines some constants which are used for creating a uniquely named directory to contain all the outputs from running this instance of this notebook. The naming scheme is based on the time that the notebook is run. The other constants are used for specifying information in the output table about what the topic was for this notebook when it was run, such as looking at KEGG biochemical pathways or STRING protein-protein interaction data some other type of gene function grouping or hierarchy. These values are arbitrary and are just for keeping better notes about what the output of the notebook corresponds to. All the input and output file paths for loading datasets or models are also contained within this cell, so that if anything is moved the directories and file names should only have to be changed at this point and nowhere else further into the notebook. If additional files are added to the notebook cells they should be put here as well.

# In[29]:


# The summarizing output dictionary has the shape TABLE[method][metric] --> value.
TOPIC = "Biochemical Pathways"
DATA = "Filtered"
TABLE = defaultdict(dict)
OUTPUT_DIR = os.path.join("../outputs",datetime.datetime.now().strftime('%m_%d_%Y_h%Hm%Ms%S'))
os.mkdir(OUTPUT_DIR)


# In[30]:


dataset_filename = "../data/pickles/text_plus_annotations_dataset.pickle"        # The full dataset pickle.
groupings_filename = "../data/pickles/kegg_pathways.pickle"                      # The groupings pickle.
background_corpus_filename = "../data/corpus_related_files/untagged_text_corpora/background.txt"       # Text file with background content.
phenotypes_corpus_filename = "../data/corpus_related_files/untagged_text_corpora/phenotypes_small.txt" # Text file with specific content.
doc2vec_pubmed_filename = "../gensim/pubmed_dbow/doc2vec_2.bin"                  # File holding saved Doc2Vec model.
doc2vec_wikipedia_filename = "../gensim/enwiki_dbow/doc2vec.bin"                 # File holding saved Doc2Vec model.
word2vec_model_filename = "../gensim/wiki_sg/word2vec.bin"                       # File holding saved Word2Vec model.
ontology_filename = "../ontologies/mo.obo"                                       # Ontology file in OBO format.
noblecoder_jarfile_path = "../lib/NobleCoder-1.0.jar"                            # Jar for NOBLE Coder tool.
biobert_pmc_path = "../gensim/biobert_v1.0_pmc/pytorch_model"                    # Path for PyTorch BioBERT model.
biobert_pubmed_path = "../gensim/biobert_v1.0_pubmed/pytorch_model"              # Path for PyTorch BioBERT model.
biobert_pubmed_pmc_path = "../gensim/biobert_v1.0_pubmed_pmc/pytorch_model"      # Path for PyTorch BioBERT model.


# <a id="2"></a>
# ### Reading in the dataset of genes and their associated phenotype descriptions and annotations

# In[32]:


dataset = load_from_pickle(dataset_filename)
dataset.describe()
dataset.filter_by_species("ath")
dataset.filter_has_description()
dataset.filter_has_annotation()
dataset.describe()
dataset.filter_has_annotation("GO")
dataset.filter_has_annotation("PO")
dataset.describe()
dataset.to_pandas().head(10)


# <a id="3"></a>
# ### Reading in the dataset of groupings, pathways, or some other kind of categorization

# In[25]:


groups = load_from_pickle(groupings_filename)
id_to_group_ids = groups.get_id_to_group_ids_dict(dataset.get_gene_dictionary())
group_id_to_ids = groups.get_group_id_to_ids_dict(dataset.get_gene_dictionary())
group_mapped_ids = [k for (k,v) in id_to_group_ids.items() if len(v)>0]
groups.describe()
groups.to_csv(os.path.join(OUTPUT_DIR,"groupings.csv"))
groups.to_pandas().head(10)


# <a id="4"></a>
# ### Relating the dataset of genes to the dataset of groupings or categories
# This section generates tables that indicate how the genes present in the dataset were mapped to the defined pathways or groups. This includes a summary table that indicates how many genes by species were succcessfully mapped to atleast one pathway or group, as well as a more detailed table describing how many genes from each species were mapped to each particular pathway or group.

# In[26]:


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
table.to_csv(os.path.join(OUTPUT_DIR,"mappings_summary.csv"), index=False)

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
table.to_csv(os.path.join(OUTPUT_DIR,"mappings_by_group.csv"), index=False)


# ### Option 1: Filtering the dataset based on presence in the curated Oellrich, Walls et al. (2015) dataset

# In[33]:


# Filter the dataset based on whether or not the genes were in the curated dataset.
# This is similar to filtering based on protein interaction data because the dataset is a list of edge values.
pppn_edgelist_path = "../data/supplemental_files_oellrich_walls/13007_2015_53_MOESM9_ESM.txt"
pppn_edgelist = Edges(dataset.get_name_to_id_dictionary(), pppn_edgelist_path)
dataset.filter_with_ids(pppn_edgelist.ids)
dataset.describe()




# ### Option 3: Filtering the dataset based on membership in pathways or phenotype category
# This is done to only include genes (and the corresponding phenotype descriptions and annotations) which are useful for the current analysis. In this case we want to only retain genes that are mapped to atleast one pathway in whatever the source of pathway membership we are using is (KEGG, Plant Metabolic Network, etc). This is because for these genes, it will be impossible to correctly predict their pathway membership, and we have no evidence that they belong or do not belong in certain pathways so they can not be identified as being true or false negatives in any case.

# In[28]:


# Filter based on succcessful mappings to groups or pathways.
dataset.filter_with_ids(group_mapped_ids)
dataset.describe()
# Get the mappings in each direction again now that the dataset has been subset.
id_to_group_ids = groups.get_id_to_group_ids_dict(dataset.get_gene_dictionary())
group_id_to_ids = groups.get_group_id_to_ids_dict(dataset.get_gene_dictionary())


# ### Loading trained and saved NLP neural network models

# In[ ]:


# Files and models related to the machine learning text embedding methods.
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


# ### Preprocessing text descriptions
# Normalizing case, lemmatization, stemming, removing stopwords, removing punctuation, handling numerics, creating parse trees, part-of-speech tagging, and anything else necessary for a particular dataset of descripitons.

# In[ ]:


# Constructing a vocabulary by looking at what words are overrepresented in domain specific text.
background_corpus = open(background_corpus_filename,"r").read()
phenotypes_corpus = open(phenotypes_corpus_filename,"r").read()
tokens = get_overrepresented_tokens(phenotypes_corpus, background_corpus, max_features=5000)
vocabulary_from_text = build_vocabulary_from_tokens(tokens)

# Constructing a vocabulary by assuming all words present in a given ontology are important.
ontology = Ontology(ontology_filename)
vocabulary_from_ontology = build_vocabulary_from_tokens(ontology.get_tokens())


# ### Preprocessing text descriptions
# The preprocessing methods applied to the phenotype descriptions are a choice which impacts the subsequent vectorization and similarity methods which construct the pairwise distance matrix from each of these descriptions. The preprocessing methods that make sense are also highly dependent on the vectorization method or embedding method that is to be applied. For example, stemming (which is part of the full proprocessing done below using `gensim.preprocess_string` is useful for the n-grams and bag-of-words methods but not for the document embeddings methods which need each token to be in the vocabulary that was constructed and used when the model was trained. For this reason, embedding methods with pretrained models where the vocabulary is fixed should have a lighter degree of preprocessing not involving stemming or lemmatization but should involve things like removal of non-alphanumerics and normalizing case. 

# In[ ]:


# Obtain a mapping between IDs and the raw text descriptions associated with that ID from the dataset.
descriptions = dataset.get_description_dictionary()

# Preprocessing of the text descriptions. Different methods are necessary for different approaches.
descriptions_full_preprocessing = {i:" ".join(preprocess_string(d)) for i,d in descriptions.items()}
descriptions_simple_preprocessing = {i:" ".join(simple_preprocess(d)) for i,d in descriptions.items()}
descriptions_no_stopwords = {i:remove_stopwords(d) for i,d in descriptions.items()}

# Generating random descriptions that are drawn from same token population and retain original lengths.
#tokens = [w for w in itertools.chain.from_iterable(word_tokenize(desc) for desc in descriptions.values())]
#descriptions_scrambled = {k:" ".join(np.random.choice(tokens,len(word_tokenize(v)))) for k,v in descriptions.items()}


# ### POS tagging the phenotype descriptions for nouns and adjectives
# Note that preprocessing of the descriptions should be done after part-of-speech tagging, because tokens that are removed during preprocessing before n-gram analysis contain information that the parser needs to accurately call parts-of-speech. This step should be done on the raw descriptions and then the resulting bags of words can be subset using additional preprocesssing steps before input in one of the vectorization methods.

# In[ ]:


get_pos_tokens = lambda text,pos: " ".join([t[0] for t in nltk.pos_tag(word_tokenize(text)) if t[1].lower()==pos.lower()])
descriptions_noun_only =  {i:get_pos_tokens(d,"NN") for i,d in descriptions.items()}
descriptions_noun_only_full_preprocessing = {i:" ".join(preprocess_string(d)) for i,d in descriptions_noun_only.items()}
descriptions_noun_only_simple_preprocessing = {i:" ".join(simple_preprocess(d)) for i,d in descriptions_noun_only.items()}
descriptions_adj_only =  {i:get_pos_tokens(d,"JJ") for i,d in descriptions.items()}
descriptions_adj_only_full_preprocessing = {i:" ".join(preprocess_string(d)) for i,d in descriptions_adj_only.items()}
descriptions_adj_only_simple_preprocessing = {i:" ".join(simple_preprocess(d)) for i,d in descriptions_adj_only.items()}


# ### Annotating descriptions with ontology terms
# This section generates dictionaries that map gene IDs from the dataset to lists of strings, where those strings are ontology term IDs. How the term IDs are found for each gene entry with its corresponding phenotype description depends on the cell below. Firstly, the terms are found by using the NOBLE Coder annotation tool through these wrapper functions to identify the terms by looking for instances of the term's label or synonyms in the actual text of the phenotype descriptions. Secondly, the next cell just draws the terms directly from the dataset itself. In this case, these are high-confidence annotations done by curators for a comparison against what can be accomplished through computational analysis of the text.

# In[ ]:


# Run the ontology term annotators over either raw or preprocessed text descriptions.
annotations_noblecoder_precise = annotate_using_noble_coder(descriptions, noblecoder_jarfile_path, "mo", precise=1)
annotations_noblecoder_partial = annotate_using_noble_coder(descriptions, noblecoder_jarfile_path, "mo", precise=0)


# In[ ]:


# Get the ID to term list annotation dictionaries for each ontology in the dataset.
annotations = dataset.get_annotations_dictionary()
go_annotations = {k:[term for term in v if term[0:2]=="GO"] for k,v in annotations.items()}
po_annotations = {k:[term for term in v if term[0:2]=="PO"] for k,v in annotations.items()}


# ### Generating vector representations and pairwise distances matrices
# This section uses the text descriptions, preprocessed text descriptions, or ontology term annotations created or read in the previous sections to generate a vector representation for each gene and build a pairwise distance matrix for the whole dataset. Each method specified is a unique combination of a method of vectorization (bag-of-words, n-grams, document embedding model, etc) and distance metric (Euclidean, Jaccard, cosine, etc) applied to those vectors in constructing the pairwise matrix. The method of vectorization here is equivalent to feature selection, so the task is to figure out which type of vectors will encode features that are useful (n-grams, full words, only words from a certain vocabulary, etc).

# In[ ]:


# Define a list of tuples, each tuple will be used to build to find a matrix of pairwise distances.
# The naming scheme for methods should include both a name substring and then a hyperparameter substring
# separated by a colon. Anything after the colon will be removed from the name and put in a separate 
# column in the output table. This is so that the name column can be directly used for making figures, so
# if two hyperparameter choices are both going to be used in a figure, keep them in the name substring not
# the hyperparameter section. The required items in each tuple are:
# Index 0: name of the method
# Index 1: function to call for running this method
# Index 2: arguments to pass to that function as dictionary of keyword args
# Index 3: distance metric to apply to vectors generated with that method


name_function_args_tuples = [
    
    # Methods that use neural networks to generate embeddings.
    ("Doc2Vec Wikipedia:Size=300", pw.pairwise_doc2vec_onegroup, {"model":doc2vec_wiki_model, "object_dict":descriptions, "metric":"cosine"}, spatial.distance.cosine),
    ("Doc2Vec PubMed:Size=100", pw.pairwise_doc2vec_onegroup, {"model":doc2vec_pubmed_model, "object_dict":descriptions, "metric":"cosine"}, spatial.distance.cosine),
    ("Word2Vec Wikipedia:Size=300,Mean", pw.pairwise_word2vec_onegroup, {"model":word2vec_model, "object_dict":descriptions, "metric":"cosine", "method":"mean"}, spatial.distance.cosine),
    ("word2vec Wikipedia:Size=300,Max", pw.pairwise_word2vec_onegroup, {"model":word2vec_model, "object_dict":descriptions, "metric":"cosine", "method":"max"}, spatial.distance.cosine),
    
    ("BERT Base:Layers=2,Concatenated", pw.pairwise_bert_onegroup, {"model":bert_model_base, "tokenizer":bert_tokenizer_base, "object_dict":descriptions, "metric":"cosine", "method":"concat", "layers":2}, spatial.distance.cosine),
    ("BERT Base:Layers=3,Concatenated", pw.pairwise_bert_onegroup, {"model":bert_model_base, "tokenizer":bert_tokenizer_base, "object_dict":descriptions, "metric":"cosine", "method":"concat", "layers":3}, spatial.distance.cosine),
    ("BERT Base:Layers=4,Concatenated", pw.pairwise_bert_onegroup, {"model":bert_model_base, "tokenizer":bert_tokenizer_base, "object_dict":descriptions, "metric":"cosine", "method":"concat", "layers":4}, spatial.distance.cosine),
    ("BERT Base:Layers=2,Summed", pw.pairwise_bert_onegroup, {"model":bert_model_base, "tokenizer":bert_tokenizer_base, "object_dict":descriptions, "metric":"cosine", "method":"sum", "layers":2}, spatial.distance.cosine),
    ("BERT Base:Layers=3,Summed", pw.pairwise_bert_onegroup, {"model":bert_model_base, "tokenizer":bert_tokenizer_base, "object_dict":descriptions, "metric":"cosine", "method":"sum", "layers":3}, spatial.distance.cosine),
    ("BERT Base:Layers=4,Summed", pw.pairwise_bert_onegroup, {"model":bert_model_base, "tokenizer":bert_tokenizer_base, "object_dict":descriptions, "metric":"cosine", "method":"sum", "layers":4}, spatial.distance.cosine),
    ("BioBERT:PMC,Layers=2,Concatenated", pw.pairwise_bert_onegroup, {"model":bert_model_pmc, "tokenizer":bert_tokenizer_pmc, "object_dict":descriptions, "metric":"cosine", "method":"concat", "layers":2}, spatial.distance.cosine),
    ("BioBERT:PMC,Layers=3,Concatenated", pw.pairwise_bert_onegroup, {"model":bert_model_pmc, "tokenizer":bert_tokenizer_pmc, "object_dict":descriptions, "metric":"cosine", "method":"concat", "layers":3}, spatial.distance.cosine),
    ("BioBERT:PMC,Layers=4,Concatenated", pw.pairwise_bert_onegroup, {"model":bert_model_pmc, "tokenizer":bert_tokenizer_pmc, "object_dict":descriptions, "metric":"cosine", "method":"concat", "layers":4}, spatial.distance.cosine),
    ("BioBERT:PubMed,Layers=4,Concatenated", pw.pairwise_bert_onegroup, {"model":bert_model_pubmed, "tokenizer":bert_tokenizer_pubmed, "object_dict":descriptions, "metric":"cosine", "method":"concat", "layers":4}, spatial.distance.cosine),
    ("BioBERT:PubMed,PMC,Layers=4,Concatenated", pw.pairwise_bert_onegroup, {"model":bert_model_pubmed_pmc, "tokenizer":bert_tokenizer_pubmed_pmc, "object_dict":descriptions, "metric":"cosine", "method":"concat", "layers":4}, spatial.distance.cosine),
        
    # Methods that use variations on the ngrams approach.
    ("N-Grams:Words,1-grams,2-grams", pw.pairwise_ngrams_onegroup, {"object_dict":descriptions_full_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,2),"max_features":5000, "tfidf":False}, spatial.distance.cosine),
    ("N-Grams:Words,1-grams,2-grams,Binary", pw.pairwise_ngrams_onegroup, {"object_dict":descriptions_full_preprocessing, "metric":"cosine", "binary":True, "analyzer":"word", "ngram_range":(1,2), "max_features":5000, "tfidf":False}, spatial.distance.cosine),
    ("N-Grams:Words,1-grams", pw.pairwise_ngrams_onegroup, {"object_dict":descriptions_full_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":5000, "tfidf":False}, spatial.distance.cosine),
    ("N-Grams:Words,1-grams,Binary", pw.pairwise_ngrams_onegroup, {"object_dict":descriptions_full_preprocessing, "metric":"jaccard", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":1000, "tfidf":False}, spatial.distance.jaccard),
    ("N-Grams:Words,1-grams,2-grams,TFIDF", pw.pairwise_ngrams_onegroup, {"object_dict":descriptions_full_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,2),"max_features":5000, "tfidf":True}, spatial.distance.cosine),
    ("N-Grams:Words,1-grams,2-grams,Binary,TFIDF", pw.pairwise_ngrams_onegroup, {"object_dict":descriptions_full_preprocessing, "metric":"cosine", "binary":True, "analyzer":"word", "ngram_range":(1,2), "max_features":5000, "tfidf":True}, spatial.distance.cosine),
    ("N-Grams:Words,1-grams,TFIDF", pw.pairwise_ngrams_onegroup, {"object_dict":descriptions_full_preprocessing, "metric":"cosine", "binary":False, "analyzer":"word", "ngram_range":(1,1), "max_features":5000, "tfidf":True}, spatial.distance.cosine),
    ("N-Grams:Words,1-grams,Binary,TFIDF", pw.pairwise_ngrams_onegroup, {"object_dict":descriptions_full_preprocessing, "metric":"jaccard", "binary":True, "analyzer":"word", "ngram_range":(1,1), "max_features":5000, "tfidf":True}, spatial.distance.jaccard),
    
    # Methods that use terms inferred from automated annotation of the text.
    ("NOBLE Coder:Precise", pw.pairwise_annotations_onegroup, {"annotations_dict":annotations_noblecoder_precise, "ontology":ontology, "binary":True, "metric":"jaccard", "tfidf":False}, spatial.distance.jaccard),
    ("NOBLE Coder:Partial", pw.pairwise_annotations_onegroup, {"annotations_dict":annotations_noblecoder_partial, "ontology":ontology, "binary":True, "metric":"jaccard", "tfidf":False}, spatial.distance.jaccard),
    ("NOBLE Coder:Precise,TFIDF", pw.pairwise_annotations_onegroup, {"annotations_dict":annotations_noblecoder_precise, "ontology":ontology, "binary":True, "metric":"jaccard", "tfidf":True}, spatial.distance.jaccard),
    ("NOBLE Coder:Partial,TFIDF", pw.pairwise_annotations_onegroup, {"annotations_dict":annotations_noblecoder_partial, "ontology":ontology, "binary":True, "metric":"jaccard", "tfidf":True}, spatial.distance.jaccard),
    
    # Methods that use terms assigned by humans that are present in the dataset.
    ("GO:Default", pw.pairwise_annotations_onegroup, {"annotations_dict":go_annotations, "ontology":ontology, "metric":"jaccard", "binary":True, "analyzer":"word", "ngram_range":(1,1), "tfidf":False}, spatial.distance.jaccard),
    ("PO:Default", pw.pairwise_annotations_onegroup, {"annotations_dict":po_annotations, "ontology":ontology, "metric":"jaccard", "binary":True, "analyzer":"word", "ngram_range":(1,1), "tfidf":False}, spatial.distance.jaccard),
]


# In[ ]:


# Generate all of the similarity matrices in parallel.
#start_time_mp = time.perf_counter()
#pool = mp.Pool(mp.cpu_count())
#results = [pool.apply_async(function_wrapper_with_duration, args=(func, args)) for (name,func,args,metric) in name_function_args_tuples]
#results = [result.get() for result in results]
#graphs = {tup[0]:result[0] for tup,result in zip(name_function_args_tuples,results)}
#metric_dict = {tup[0]:tup[3] for tup in name_function_args_tuples}
#durations = {tup[0]:result[1] for tup,result in zip(name_function_args_tuples,results)}
#pool.close()
#pool.join()    
#total_time_mp = time.perf_counter()-start_time_mp

# Reporting how long each matrix took to build and how much time parallel processing saved.
#print("Durations of generating each pairwise similarity matrix (hh:mm:ss)")
#print("-----------------------------------------------------------------")
#savings = total_time_mp/sum(durations.values())
#for (name,duration) in durations.items():
#    print("{:15} {}".format(name, to_hms(duration)))
#print("-----------------------------------------------------------------")
#print("{:15} {}".format("total", to_hms(sum(durations.values()))))
#print("{:15} {} ({:.2%} of single thread time)".format("multiprocess", to_hms(total_time_mp), savings))


# In[ ]:


# Generate all the pairwise distance matrices.
graphs = {tup[0]:tup[1](**tup[2]) for tup in name_function_args_tuples}
metric_dict = {tup[0]:tup[3] for tup in name_function_args_tuples}


# In[ ]:


# Merging all of the edgelist dataframes together.
methods = list(graphs.keys())
edgelists = {k:v.edgelist for k,v in graphs.items()}
df = pw.merge_edgelists(edgelists, default_value=0.000)
df = pw.remove_self_loops(df)
df.tail(10)


# ### Option 1: Merging in the previously curated similarity values from the Oellrich, Walls et al. (2015) dataset
# This section reads in a file that contains the previously calculated distance values from the Oellrich, Walls et al. (2015) dataset, and merges it with the values which are obtained here for all of the applicable natural language processing or machine learning methods used, so that the graphs which are specified by these sets of distances values can be evaluated side by side in the subsequent sections.

# In[ ]:


# Add a column that indicates the distance estimated using curated EQ statements.
df = df.merge(right=pppn_edgelist.df, how="left", on=["from","to"])
df.fillna(value=0.000,inplace=True)
df.rename(columns={"value":"Curated"}, inplace=True)
df["Curated"] = 1-df["Curated"]
methods.append("Curated")
df.tail(10)


# ### Option 2: Merging with information about shared pathways or groupings.
# The relevant information for each edge includes questions like whether or not the two genes that edge connects share a group or biochemical pathway in common, or if those genes are from the same species. This information can then later be used as the target values for predictive models, or for filtering the graphs represented by these edge lists. Either the grouping information or the protein-protein interaction information should be used.

# In[ ]:


# Generate a column indicating whether or not the two genes have atleast one pathway in common.
df["shared"] = df[["from","to"]].apply(lambda x: len(set(id_to_group_ids[x["from"]]).intersection(set(id_to_group_ids[x["to"]])))>0, axis=1)*1
print(Counter(df["shared"].values))

# Generate a column indicating whether or not the two genes are from the same species.
species_dict = dataset.get_species_dictionary()
df["same"] = df[["from","to"]].apply(lambda x: species_dict[x["from"]]==species_dict[x["to"]],axis=1)*1
print(Counter(df["same"].values))


# ### Combining multiple distance methods with machine learning models
# The purpose of this section is to iteratively train models on subsections of the dataset using simple regression or machine learning approaches to predict a value from zero to one indicating indicating how likely is it that two genes share atleast one of the specified groups in common. The information input to these models is the distance scores provided by each method in some set of all the methods used in this notebook. The purpose is to see whether or not a function of these similarity scores specifically trained to the task of predicting common groupings is better able to used the distance metric information to report a score for this task.

# In[ ]:


"""
# Iteratively create models for combining output values from multiple semantic similarity methods.
method = "Logistic Regression"
splits = 12
kf = KFold(n_splits=splits, random_state=14271, shuffle=True)
df[method] = pd.Series()
for train,test in kf.split(df):
    lr_model = train_logistic_regression_model(df=df.iloc[train], predictor_columns=methods, target_column="shared")
    df[method].iloc[test] = apply_logistic_regression_model(df=df.iloc[test], predictor_columns=methods, model=lr_model)
df[method] = 1-df[method]
methods.append(method)
"""


# In[ ]:


"""
# Iteratively create models for combining output values from multiple semantic similarity methods.
method = "Random Forest"
splits = 2
kf = KFold(n_splits=splits, random_state=14271, shuffle=True)
df[method] = pd.Series()
for train,test in kf.split(df):
    rf_model = train_random_forest_model(df=df.iloc[train], predictor_columns=methods, target_column="shared")
    df[method].iloc[test] = apply_random_forest_model(df=df.iloc[test],predictor_columns=methods, model=rf_model)
df[method] = 1-df[method]
methods.append(method)
"""


# In[ ]:


"""
# Don't do it iteratively, just treat this entire dataset as training data and save the models.
# Logistic Regression
lr_name = "Logistic Regression"
model = train_logistic_regression_model(df=df, predictor_columns=methods, target_column="shared")
df[lr_name] = apply_logistic_regression_model(df=df, predictor_columns=methods, model=model)
df[lr_name] = 1-df[lr_name]
save_to_pickle(obj=model,path="../data/pickles/lr.model")
# Random Forest
rf_name = "Random Forest"
model = train_random_forest_model(df=df, predictor_columns=methods, target_column="shared")
df[rf_name] = apply_random_forest_model(df=df, predictor_columns=methods, model=model)
df[rf_name] = 1-df[rf_name]
save_to_pickle(obj=model,path="../data/pickles/rf.model")
# Add these methods to the method names to use for the below analysis.
methods.append(lr_name)
methods.append(rf_name)
"""


# ### Do the edges joining genes that share atleast one pathway come from a different distribution?
# The purpose of this section is to visualize kernel estimates for the distributions of distance or similarity scores generated by each of the methods tested for measuring semantic similarity or generating vector representations of the phenotype descriptions. Ideally, better methods should show better separation betwene the distributions for distance values between two genes involved in a common specified group or two genes that are not. Additionally, a statistical test is used to check whether these two distributions are significantly different from each other or not, although this is a less informative measure than the other tests used in subsequent sections, because it does not address how useful these differences in the distributions actually are for making predictions about group membership.

# In[ ]:


# Use Kolmogorov-Smirnov test to see if edges between genes that share a group come from a distinct distribution.
ppi_pos_dict = {name:(df[df["shared"] > 0.00][name].values) for name in methods}
ppi_neg_dict = {name:(df[df["shared"] == 0.00][name].values) for name in methods}
for name in methods:
    stat,p = ks_2samp(ppi_pos_dict[name],ppi_neg_dict[name])
    pos_mean = np.average(ppi_pos_dict[name])
    neg_mean = np.average(ppi_neg_dict[name])
    pos_n = len(ppi_pos_dict[name])
    neg_n = len(ppi_neg_dict[name])
    TABLE[name].update({"mean_1":pos_mean, "mean_0":neg_mean, "n_1":pos_n, "n_0":neg_n})
    TABLE[name].update({"ks":stat, "ks_pval":p})
    
    
# Show the kernel estimates for each distribution of weights for each method.
num_plots, plots_per_row, row_width, row_height = (len(methods), 4, 14, 3)
fig,axs = plt.subplots(math.ceil(num_plots/plots_per_row), plots_per_row, squeeze=False)
for name,ax in zip(methods,axs.flatten()):
    ax.set_title(name)
    ax.set_xlabel("value")
    ax.set_ylabel("density")
    sns.kdeplot(ppi_pos_dict[name], color="black", shade=False, alpha=1.0, ax=ax)
    sns.kdeplot(ppi_neg_dict[name], color="black", shade=True, alpha=0.1, ax=ax) 
fig.set_size_inches(row_width, row_height*math.ceil(num_plots/plots_per_row))
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR,"kernel_density.png"),dpi=400)
plt.close()


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
for method in methods:
    all_weights_dict[method] = df[method].values
    for group in group_ids:
        within_ids = group_id_to_ids[group]
        within_pairs = [(i,j) for i,j in itertools.permutations(within_ids,2)]
        mean_weight = np.mean((graph.get_values(within_pairs, kind=method)))
        within_weights_dict[method][group] = mean_weight
        within_percentiles_dict[method][group] = stats.percentileofscore(df[method].values, mean_weight, kind="rank")

# Generating a dataframe of percentiles of the mean in-group distance scores.
heatmap_data = pd.DataFrame(within_percentiles_dict)
heatmap_data = heatmap_data.dropna(axis=0, inplace=False)
heatmap_data = heatmap_data.round(4)

# Adding relevant information to this dataframe and saving.
heatmap_data["mean_rank"] = heatmap_data.rank().mean(axis=1)
heatmap_data["mean_percentile"] = heatmap_data.mean(axis=1)
heatmap_data.sort_values(by="mean_percentile", inplace=True)
heatmap_data.reset_index(inplace=True)
heatmap_data["group_id"] = heatmap_data["index"]
heatmap_data["full_name"] = heatmap_data["group_id"].apply(lambda x: groups.get_long_name(x))
heatmap_data["n"] = heatmap_data["group_id"].apply(lambda x: len(group_id_to_ids[x]))
heatmap_data = heatmap_data[flatten(["group_id","full_name","n","mean_percentile","mean_rank",methods])]
heatmap_data.to_csv(os.path.join(OUTPUT_DIR,"within_distances.csv"), index=False)
heatmap_data.head(5)


# ### Predicting whether two genes belong to a common biochemical pathway
# The purpose of this section is to see if whether or not two genes share atleast one common pathway can be predicted from the distance scores assigned using analysis of text similarity. The evaluation of predictability is done by reporting a precision and recall curve for each method, as well as remembering the area under the curve, and ratio between the area under the curve and the baseline (expected area when guessing randomly) for each method.

# In[ ]:


y_true_dict = {name:df["shared"] for name in methods}
y_prob_dict = {name:(1 - df[name].values) for name in methods}
num_plots, plots_per_row, row_width, row_height = (len(methods), 4, 14, 3)
fig,axs = plt.subplots(math.ceil(num_plots/plots_per_row), plots_per_row, squeeze=False)
for method,ax in zip(methods, axs.flatten()):
    
    # Obtaining the values and metrics.
    y_true, y_prob = y_true_dict[method], y_prob_dict[method]
    n_pos, n_neg = Counter(y_true)[1], Counter(y_true)[0]
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    baseline = Counter(y_true)[1]/len(y_true) 
    area = auc(recall, precision)
    auc_to_baseline_auc_ratio = area/baseline
    TABLE[method].update({"auc":area, "baseline":baseline, "ratio":auc_to_baseline_auc_ratio})

    # Producing the precision recall curve.
    step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
    ax.step(recall, precision, color='black', alpha=0.2, where='post')
    ax.fill_between(recall, precision, alpha=0.7, color='black', **step_kwargs)
    ax.axhline(baseline, linestyle="--", color="lightgray")
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.set_title("PR {0} (Baseline={1:0.3f})".format(method, baseline))
    
fig.set_size_inches(row_width, row_height*math.ceil(num_plots/plots_per_row))
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR,"prcurve_shared.png"),dpi=400)
plt.close()


# ### Are genes in the same biochemical pathway ranked higher with respect to individual nodes?
# This is a way of statistically seeing if for some value k, the graph ranks more edges from some particular gene to any other gene that it has a true protein-protein interaction with higher or equal to rank k, than we would expect due to random chance. This way of looking at the problem helps to be less ambiguous than the previous methods, because it gets at the core of how this would actually be used. In other words, we don't really care how much true information we're missing as long as we're still able to pick up some new useful information by building these networks, so even though we could be missing a lot, what's going on at the very top of the results? These results should be comparable to very strictly thresholding the network and saying that the remaining edges are our guesses at interactions. This is comparable to just looking at the far left-hand side of the precision recall curves, but just quantifies it slightly differently.

# In[ ]:


# When the edgelist is generated above, only the lower triangle of the pairwise matrix is retained for edges in the 
# graph. This means that in terms of the indices of each node, only the (i,j) node is listed in the edge list where
# i is less than j. This makes sense because the graph that's specified is assumed to already be undirected. However
# in order to be able to easily subset the edgelist by a single column to obtain rows that correspond to all edges
# connected to a particular node, this method will double the number of rows to include both (i,j) and (j,i) edges.
df = pw.make_undirected(df)

# What's the number of functional partners ranked k or higher in terms of phenotypic description similarity for 
# each gene? Also figure out the maximum possible number of functional partners that could be theoretically
# recovered in this dataset if recovered means being ranked as k or higher here.
k = 10      # The threshold of interest for gene ranks.
n = 100     # Number of Monte Carlo simulation iterations to complete.
df[list(methods)] = df.groupby("from")[list(methods)].rank()
ys = df[df["shared"]==1][list(methods)].apply(lambda s: len([x for x in s if x<=k]))
ymax = sum(df.groupby("from")["shared"].apply(lambda s: min(len([x for x in s if x==1]),k)))

# Monte Carlo simulation to see what the probability is of achieving each y-value by just randomly pulling k 
# edges for each gene rather than taking the top k ones that the similarity methods specifies when ranking.
ysims = [sum(df.groupby("from")["shared"].apply(lambda s: len([x for x in s.sample(k) if x>0.00]))) for i in range(n)]
for method in methods:
    pvalue = len([ysim for ysim in ysims if ysim>=ys[method]])/float(n)
    TABLE[method].update({"y":ys[method], "y_max":ymax, "y_ratio":ys[method]/ymax, "y_pval":pvalue})


# ### Predicting biochemical pathway or group membership based on mean vectors
# This section looks at how well the biochemical pathways that a particular gene is a member of can be predicted based on the similarity between the vector representation of the phenotype descriptions for that gene and the average vector for all the vector representations of phenotypes asociated with genes that belong to that particular pathway. In calculating the average vector for a given biochemical pathway, the vector corresponding to the gene that is currently being classified is not accounted for, to avoid overestimating the performance by including information about the ground truth during classification. This leads to missing information in the case of biochemical pathways that have only one member. This can be accounted for by only limiting the overall dataset to only include genes that belong to pathways that have atleast two genes mapped to them, and only including those pathways, or by removing the missing values before calculating the performance metrics below.

# In[ ]:


# Get the list of methods to look at, and a mapping between each method and the correct similarity metric to apply.
vector_dicts = {k:v.vector_dictionary for k,v in graphs.items()}
methods = list(vector_dicts.keys())
group_id_to_ids = groups.get_group_id_to_ids_dict(dataset.get_gene_dictionary())
valid_group_ids = [group for group,id_list in group_id_to_ids.items() if len(id_list)>1]
valid_ids = [i for i in dataset.get_ids() if len(set(valid_group_ids).intersection(set(id_to_group_ids[i])))>0]
pred_dict = defaultdict(lambda: defaultdict(dict))
true_dict = defaultdict(lambda: defaultdict(dict))
for method in methods:
    for group in valid_group_ids:
        ids = group_id_to_ids[group]
        for identifier in valid_ids:
            # What's the mean vector of this group, without this particular one that we're trying to classify.
            vectors = np.array([vector_dicts[method][some_id] for some_id in ids if not some_id==identifier])
            mean_vector = vectors.mean(axis=0)
            this_vector = vector_dicts[method][identifier]
            pred_dict[method][identifier][group] = 1-metric_dict[method](mean_vector, this_vector)
            true_dict[method][identifier][group] = (identifier in group_id_to_ids[group])*1                


# In[ ]:


num_plots, plots_per_row, row_width, row_height = (len(methods), 4, 14, 3)
fig,axs = plt.subplots(math.ceil(num_plots/plots_per_row), plots_per_row, squeeze=False)
for method,ax in zip(methods, axs.flatten()):
    
    # Obtaining the values and metrics.
    y_true = pd.DataFrame(true_dict[method]).as_matrix().flatten()
    y_prob = pd.DataFrame(pred_dict[method]).as_matrix().flatten()
    n_pos, n_neg = Counter(y_true)[1], Counter(y_true)[0]
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    baseline = Counter(y_true)[1]/len(y_true) 
    area = auc(recall, precision)
    auc_to_baseline_auc_ratio = area/baseline
    TABLE[method].update({"mean_auc":area, "mean_baseline":baseline, "mean_ratio":auc_to_baseline_auc_ratio})

    # Producing the precision recall curve.
    step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
    ax.step(recall, precision, color='black', alpha=0.2, where='post')
    ax.fill_between(recall, precision, alpha=0.7, color='black', **step_kwargs)
    ax.axhline(baseline, linestyle="--", color="lightgray")
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.set_title("PR {0} (Baseline={1:0.3f})".format(method[:10], baseline))
    
fig.set_size_inches(row_width, row_height*math.ceil(num_plots/plots_per_row))
fig.tight_layout()
fig.show()
fig.savefig(os.path.join(OUTPUT_DIR,"prcurve_mean_classifier.png"),dpi=400)


# ### Predicting biochemical pathway membership based on mean similarity values
# This section looks at how well the biochemical pathways that a particular gene is a member of can be predicted based on the average similarity between the vector representationt of the phenotype descriptions for that gene and each of the vector representations for other phenotypes associated with genes that belong to that particular pathway. In calculating the average similarity to other genes from a given biochemical pathway, the gene that is currently being classified is not accounted for, to avoid overestimating the performance by including information about the ground truth during classification. This leads to missing information in the case of biochemical pathways that have only one member. This can be accounted for by only limiting the overall dataset to only include genes that belong to pathways that have atleast two genes mapped to them, and only including those pathways, or by removing the missing values before calculating the performance metrics below.

# ### Predicting biochemical pathway or group membership with KNN classifier
# This section looks at how well the group(s) or biochemical pathway(s) that a particular gene belongs to can be predicted based on a KNN classifier generated using every other gene. For this section, only the groups or pathways which contain more than one gene, and the genes mapped to those groups or pathways, are of interest. This is because for other genes, if we consider them then it will be true that that gene belongs to that group in the target vector, but the KNN classifier could never predict this because when that gene is held out, nothing could provide a vote for that group, because there are zero genes available to be members of the K nearest neighbors.

# In[ ]:


"""
# Trying something new here
valid_group_ids = [group for group,id_list in group_id_to_ids.items() if len(id_list)>1]
valid_ids = [i for i in dataset.get_ids() if len(set(valid_group_ids).intersection(set(id_to_group_ids[i])))>0]

# Leave one out predictions.
pred_dict = defaultdict(lambda: defaultdict(dict))
true_dict = defaultdict(lambda: defaultdict(dict))

for method in methods[0:]:
    print(method)    
    print(vector_dicts[method][237][:20])
    print(vector_dicts[method][116][:20])
    X = [vector_dicts[method][identifier] for identifer in valid_ids]
    y = [identifier for identifier in valid_ids]
    neigh = KNeighborsClassifier(n_neighbors=4, metric="euclidean")
    neigh.fit(X,y)  
    probs = neigh.kneighbors(return_distance=True)
    for a in probs[0]:
        for b in a:
            print(float(b))
    #print(probs[0])
    print(probs[1])
    #print(len(valid_ids))
    #print(len(probs[1]))
    #print(probs[1][478])
    break
"""


# In[ ]:



"""
valid_group_ids = [group for group,id_list in group_id_to_ids.items() if len(id_list)>1]
valid_ids = [i for i in dataset.get_ids() if len(set(valid_group_ids).intersection(set(id_to_group_ids[i])))>0]

# Leave one out predictions.
pred_dict = defaultdict(lambda: defaultdict(dict))
true_dict = defaultdict(lambda: defaultdict(dict))

for method in methods:
    for i in valid_ids:
        ids = [identifier for identifier in valid_ids if identifier!=i]
        # Make the labels just the ID's for that gene instead of pathways or groups, transform votes later.
        X = [vector_dicts[method][identifier] for identifier in ids]
        y = [identifier for identifier in ids]
        neigh = KNeighborsClassifier(n_neighbors=5)
        neigh.fit(X,y)
        probs = neigh.predict_proba([vector_dicts[method][i]])
        # The classes are given in lexicographic order from the predict_proba method.
        # 1. Create a 2D binary array with a row for each group and a column for each gene.
        # 2. Multiply each row of that 2D binary array with the 1D array that has the probability for each gene.
        # 3. Find the sum of each row to get the score specific to each group for this particular gene.
        # 4. Append to the arrays for the true binary classifications and the predictions for each group. 
        ids.sort()
        binary_arr = np.array([[(group_id in id_to_group_ids[x])*1 for x in ids] for group_id in valid_group_ids]) 
        group_probs_by_gene = probs*binary_arr
        group_probs = np.sum(group_probs_by_gene, axis=1)
        for group_index,group_id in enumerate(valid_group_ids):
            pred_dict[method][i][group_id] = group_probs[group_index]
            true_dict[method][i][group_id] = (i in group_id_to_ids[group_id])*1
"""


# In[ ]:


"""
num_plots, plots_per_row, row_width, row_height = (len(methods), 4, 14, 3)
fig,axs = plt.subplots(math.ceil(num_plots/plots_per_row), plots_per_row, squeeze=False)
for method,ax in zip(methods, axs.flatten()):
    
    # Obtaining the values and metrics.
    y_true = pd.DataFrame(true_dict[method]).as_matrix().flatten()
    y_prob = pd.DataFrame(pred_dict[method]).as_matrix().flatten()
    n_pos, n_neg = Counter(y_true)[1], Counter(y_true)[0]
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    baseline = Counter(y_true)[1]/len(y_true) 
    area = auc(recall, precision)
    auc_to_baseline_auc_ratio = area/baseline
    TABLE[method].update({(TAG,"knn_auc"):area, (TAG,"knn_baseline"):baseline, (TAG,"knn_ratio"):auc_to_baseline_auc_ratio})

    # Producing the precision recall curve.
    step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
    ax.step(recall, precision, color='black', alpha=0.2, where='post')
    ax.fill_between(recall, precision, alpha=0.7, color='black', **step_kwargs)
    ax.axhline(baseline, linestyle="--", color="lightgray")
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.set_title("PR {0} (Baseline={1:0.3f})".format(method, baseline))
    
fig.set_size_inches(row_width, row_height*math.ceil(num_plots/plots_per_row))
fig.tight_layout()
fig.show()
fig.savefig(os.path.join(OUTPUT_DIR,"prcurve_knn_classifier.png"))
"""


# ### Summarizing the results for this notebook

# In[ ]:


results = pd.DataFrame(TABLE).transpose()
columns = flatten(["Hyperparams","Topic","Data",results.columns])
results["Hyperparams"] = ""
results["Topic"] = TOPIC
results["Data"] = DATA
results = results[columns]
results.reset_index(inplace=True)
results = results.rename({"index":"Method"}, axis="columns")
hyperparam_sep = ":"
results["Hyperparams"] = results["Method"].map(lambda x: x.split(hyperparam_sep)[1] if hyperparam_sep in x else "-")
results["Method"] = results["Method"].map(lambda x: x.split(hyperparam_sep)[0])
results.to_csv(os.path.join(OUTPUT_DIR,"full_table.csv"), index=True)
results

