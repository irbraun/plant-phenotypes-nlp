import sys
import os
import pandas as pd
import itertools


sys.path.append("../.")
from phenolog.Dataset import Dataset
import phenolog.nlp
import phenolog.pathway




# Read in a subset of dataframe from a TAIR public release tsv containing fields of interest.
filename = "/Users/irbraun/phenolog/data/tair/Locus_Germplasm_Phenotype_20180702.txt"
usecols = ["LOCUS_NAME", "GERMPLASM_NAME", "PHENOTYPE", "PUBMED_ID"]
usenames = ["locus", "germplasm", "description", "pubmed"]
renamed = {k:v for k,v in zip(usecols,usenames)}
df = pd.read_table(filename, usecols=usecols)
df.rename(columns=renamed, inplace=True)
df["species"] = "ath"

# Create a dataset object that can be added to.
dataset = Dataset()
dataset.add_data(df)



# Prepare a dictionary of phenotype descriptions where each has a unique ID value.
description_dict = dataset.get_description_dictionary(all_rows=1)
description_dict = {i:phenolog.nlp.get_clean_description(d) for (i,d) in description_dict.items()}



# Prepare a dictionary mapping those identifiers to locus names.
locus_dict = dataset.get_locus_dictionary(all_rows=1)








# Get mappings between pathways in KEGG and sets of related loci, and inverse as well.
pathway_dicts = phenolog.pathway.get_kegg_pathway_dictionary("ath")
pathway_dict_fwd = pathway_dicts[0]
pathway_dict_rev = pathway_dicts[1]


# Relating the two datasets to one another.
pathway_loci_set = set.union(*pathway_dict_fwd.values())
textdata_loci_set = set(locus_dict.values())
intersection = pathway_loci_set.intersection(textdata_loci_set)
pathways_in_textdata = set(itertools.chain.from_iterable([pathway_dict_rev[loci] for loci in intersection])) 


# Some information about what was read in.
print("Number of text descriptions:", len(description_dict))
print("Number of loci in text data:", len(textdata_loci_set))
print("Number of pathways:", len(pathway_dict_fwd))
print("Number of unique loci in those pathways:", len(pathway_loci_set))
print("Number of loci in both datasets:", len(intersection))
print("NUmber of pathways represented in text data:", len(pathways_in_textdata))



