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
import itertools
from collections import Counter, defaultdict
from inspect import signature
from scipy.stats import ks_2samp
from sklearn.metrics import precision_recall_curve, f1_score, auc

sys.path.append("../../oats")
from oats.utils.utils import save_to_pickle, load_from_pickle, merge_list_dicts, flatten
from oats.datasets.dataset import Dataset
from oats.datasets.groupings import Groupings
from oats.datasets.string import get_stringdb_information
from oats.annotation.ontology import Ontology
from oats.annotation.annotation import write_annotations_to_tsv_file, read_annotations_from_tsv_file
from oats.graphs.pairwise import pairwise_edgelist_doc2vec, pairwise_edgelist_counting, pairwise_edgelist_annotations
from oats.graphs.pairwise import merge_edgelists, subset_edgelist_with_ids
from oats.graphs.pairwise import remove_self_loops

warnings.simplefilter('ignore')



string2name_file = "../data/group_related_files/string/all_organisms.name_2_string.tsv"

df = pd.read_csv(string2name_file, sep="\t")
print(df.head())
print(pd.unique(df["NCBI taxid"]))








'''
# Mapping between species codes and relevant files, used in searching pathway databases.
species_dict = {
    "ath":"../data/group_related_files/pmn/aracyc_pathways.20180702", 
    "zma":"../data/group_related_files/pmn/corncyc_pathways.20180702", 
    "mtr":"../data/group_related_files/pmn/mtruncatulacyc_pathways.20180702", 
    "osa":"../data/group_related_files/pmn/oryzacyc_pathways.20180702", 
    "gmx":"../data/group_related_files/pmn/soycyc_pathways.20180702",
    "sly":"../data/group_related_files/pmn/tomatocyc_pathways.20180702"}


# Create and save the pathways object using PMN.
pathways = Groupings(species_dict, source="pmn")
pathways.describe()
save_to_pickle(obj=pathways, path="../data/pickles/pmn_pathways.pickle")


# Create and save a pathways object using KEGG.
#pathways = Groupings(species_dict, source="kegg")
pathways.describe()
save_to_pickle(obj=pathways, path="../data/pickles/kegg_pathways.pickle")





# Some preprocessing necessary for the Lloyd three-level (group, class, subset) functional hierarchy dataset.
df = pd.read_csv("../data/group_related_files/lloyd/lloyd_gene_to_function_irb_cleaned.csv")
df.fillna("", inplace=True)
df["Alias Symbols"] = df["Alias Symbols"].apply(lambda x: other_delim_to_bar_delim(string=x, delim=";"))
df["gene_names"] = np.vectorize(concatenate_with_bar_delim)(df["Locus"], df["Gene Symbol"], df["Alias Symbols"], df["Full Gene Name"])
# Specific to classes (more general).
df_class = df[["Phenotype Classb", "gene_names"]]
df_class.columns = ["group_id", "gene_names"]
df_class.to_csv("../data/scratch/arabidopsis_classes.csv", index=False)
# Specific to subsets (more specific).
df_subset = df[["Phenotype Subsetsb", "gene_names"]]
df_subset.columns = ["group_id", "gene_names"]
df_subset["group_id"] = df_subset["group_id"].apply(lambda x: x.replace("W:", "").replace("S:",""))
df_subset["group_id"] = df_subset["group_id"].apply(lambda x: other_delim_to_bar_delim(string=x, delim=","))
df_subset.to_csv("../data/scratch/arabidopsis_subsets.csv", index=False)
# Provide a mapping from subset or class IDs to the longer names that define them.
df = pd.read_csv("../data/group_related_files/lloyd/lloyd_function_hierarchy_irb_cleaned.csv")
subset_id_to_name_dict = {row[5]:row[7] for row in df.itertuples()}
class_id_to_name_dict = {row[3]:row[4] for row in df.itertuples()}
print(class_id_to_name_dict)



# Create the save the grouping object using Lloyd function hierarchy dataset of subsets.
species_dict = {"ath":"../data/scratch/arabidopsis_subsets.csv"}
subsets = Groupings(species_dict, source="csv", name_mapping=subset_id_to_name_dict)
subsets.describe()
save_to_pickle(obj=subsets, path="../data/pickles/lloyd_subsets.pickle")

# Create the save the grouping object using Lloyd function hierarchy dataset of classes.
species_dict = {"ath":"../data/scratch/arabidopsis_classes.csv"}
classes = Groupings(species_dict, source="csv", name_mapping=class_id_to_name_dict)
classes.describe()
save_to_pickle(obj=classes, path="../data/pickles/lloyd_classes.pickle")
'''