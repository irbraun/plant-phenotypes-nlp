import sys
import pandas as pd
import numpy as np

from oats.datasets.groupings import Groupings
from oats.utils.utils import save_to_pickle, load_from_pickle
from oats.nlp.preprocess import other_delim_to_bar_delim, concatenate_with_bar_delim




# Mapping between species codes and relevant files, used in searching pathway databases.
species_dict = {
    "ath":"../data/pathways/plantcyc/aracyc_pathways.20180702", 
    "zma":"../data/pathways/plantcyc/corncyc_pathways.20180702", 
    "mtr":"../data/pathways/plantcyc/mtruncatulacyc_pathways.20180702", 
    "osa":"../data/pathways/plantcyc/oryzacyc_pathways.20180702", 
    "gmx":"../data/pathways/plantcyc/soycyc_pathways.20180702",
    "sly":"../data/pathways/plantcyc/tomatocyc_pathways.20180702"}


# Create and save the pathways object using PMN.
pathways = Groupings(species_dict, source="pmn")
pathways.describe()
save_to_pickle(obj=pathways, path="../data/pickles/pmn_pathways.pickle")


# Create and save a pathways object using KEGG.
pathways = Groupings(species_dict, source="kegg")
pathways.describe()
save_to_pickle(obj=pathways, path="../data/pickles/kegg_pathways.pickle")








# Some preprocessing necessary for the Lloyd three-level (group, class, subset) functional hierarchy dataset.
df = pd.read_csv("../data/sources/cleaned/lloyd_gene_to_function.csv")
df.fillna("", inplace=True)
df["Alias Symbols"] = df["Alias Symbols"].apply(lambda x: other_delim_to_bar_delim(string=x, delim=";"))
df["gene_names"] = np.vectorize(concatenate_with_bar_delim)(df["Locus"], df["Gene Symbol"], df["Alias Symbols"], df["Full Gene Name"])
# Specific to classes (more general).
df_class = df[["Phenotype Classb", "gene_names"]]
df_class.columns = ["group_id", "gene_names"]
df_class.to_csv("../data/reshaped/arabidopsis_classes.csv", index=False)
# Specific to subsets (more specific).
df_subset = df[["Phenotype Subsetsb", "gene_names"]]
df_subset.columns = ["group_id", "gene_names"]
df_subset["group_id"] = df_subset["group_id"].apply(lambda x: x.replace("W:", "").replace("S:",""))
df_subset["group_id"] = df_subset["group_id"].apply(lambda x: other_delim_to_bar_delim(string=x, delim=","))
df_subset.to_csv("../data/reshaped/arabidopsis_subsets.csv", index=False)





# Create the save the grouping object using Lloyd function hierarchy dataset of subsets.
species_dict = {"ath":"../data/reshaped/arabidopsis_subsets.csv"}
subsets = Groupings(species_dict, source="csv")
subsets.describe()
save_to_pickle(obj=subsets, path="../data/pickles/lloyd_subsets.pickle")


# Create the save the grouping object using Lloyd function hierarchy dataset of classes.
species_dict = {"ath":"../data/reshaped/arabidopsis_classes.csv"}
classes = Groupings(species_dict, source="csv")
classes.describe()
save_to_pickle(obj=classes, path="../data/pickles/lloyd_classes.pickle")