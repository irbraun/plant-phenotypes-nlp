import sys
import os
import pandas as pd
import itertools


sys.path.append("../.")
from phenolog.Dataset import Dataset
import phenolog.nlp
import phenolog.pathway




# Read in a subset of dataframe from a TAIR public release tsv containing fields of interest.
filename = "/Users/irbraun/phenolog/data/maizegdb/maizegdb_locus_phenotype_data_2019.csv"




usecols = ["locus_name", "phenotype_description"]
usenames = ["locus", "description"]
renamed = {k:v for k,v in zip(usecols,usenames)}
df = pd.read_csv(filename, usecols=usecols)
df.rename(columns=renamed, inplace=True)
df["pubmed"] = "unknown"
df["species"] = "zma"

# Create a dataset object that can be added to.
dataset = Dataset()
dataset.add_data(df)


# Prepare a dictionary of phenotype descriptions where each has a unique ID value.
description_dict = dataset.get_description_dictionary(all_rows=1)
description_dict = {i:phenolog.nlp.get_clean_description(d) for (i,d) in description_dict.items()}



# Prepare a dictionary mapping those identifiers to locus names.
locus_dict = dataset.get_locus_dictionary(all_rows=1)





# Get mappings between pathways in KEGG and sets of related loci, and inverse as well.
df = phenolog.pathway.get_kegg_pathway_dataframe("zma") 

print(df.head(30))





'''
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
'''