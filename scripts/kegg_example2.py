import sys
import os
import pandas as pd
import numpy as np
import itertools


sys.path.append("../.")
from phenolog.Dataset import Dataset
import phenolog.nlp
from phenolog.Pathways import Pathways












# Read in a subset of dataframe from a TAIR public release tsv containing fields of interest.
filename = "/Users/irbraun/phenolog/data/maizegdb/maizegdb_locus_phenotype_data_2019.csv"


usecols = ["locus_name", "phenotype_description"]
usenames = ["gene_names", "description"]
renamed = {k:v for k,v in zip(usecols,usenames)}
df = pd.read_csv(filename, usecols=usecols)
df.rename(columns=renamed, inplace=True)
df["pubmed"] = "unknown"
df["species"] = "zma"
df["gene_ncbi"] = ""
df["gene_uniprot"] = ""


# Create a dataset object that can be added to.
dataset = Dataset()
dataset.add_data(df)


# Prepare a dictionary of phenotype descriptions where each has a unique ID value.
description_dict = dataset.get_description_dictionary(all_rows=1)
description_dict = {i:phenolog.nlp.get_clean_description(d) for (i,d) in description_dict.items()}


# Prepare a dictionary mapping those identifiers to locus names.
gene_dict = dataset.get_gene_dictionary()



# Generate dataframes from KEGG information for each species.
species_codes = ["ath", "zma", "mtr", "osa", "gmx", "sly"]
species_codes = ["zma"]
pathways = Pathways(species_codes)






d = pathways.get_kegg_pathway_dict(species="zma", gene_dict=gene_dict)


num_pathways_by_gene = [len(pathway_list) for pathway_list in d.values()]
print(np.histogram(num_pathways_by_gene, density=False))








#print(d)
print("done")
sys.exit()






# Print out the information found for each species.
for species_code in species_codes:
	df = species_to_df[species_code]
	print("Species:", species_code)
	print("Number of pathways:", len(pd.unique(df.pathway_id)))
	print("Number of unique genes in those pathways:", len(pd.unique(df.pathway_id)))
	print("N")









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