import sys
import os
import pandas as pd
import numpy as np
import itertools


sys.path.append("../.")
from phenolog.Dataset import Dataset
from phenolog.Pathways import Pathways
import phenolog.nlp
import phenolog.utils







# Create a dataset of the phenotype descriptions and corresponding gene information.
dataset = Dataset()
dataset.add_data(pd.read_csv("../data/reshaped/arabidopsis_phenotypes.csv", lineterminator="\n"))
dataset.add_data(pd.read_csv("../data/reshaped/maize_phenotypes.csv", lineterminator="\n"))
dataset.add_data(pd.read_csv("../data/reshaped/ppn_phenotypes.csv", lineterminator="\n"))
dataset.add_data(pd.read_csv("../data/reshaped/ppn_phenes.csv", lineterminator="\n"))
dataset.describe()


# Collapse the dataset so that multiple mentions of the same genes are merged.
#dataset.set_genes_as_nodes()


# Prepare a dictionary of phenotype descriptions where each has a unique ID value.
description_dict = dataset.get_description_dictionary()
description_dict = {i:phenolog.nlp.get_clean_description(d) for (i,d) in description_dict.items()}


# Prepare a dictionary mapping those ID values to gene objects with gene name information.
gene_dict = dataset.get_gene_dictionary()





# Species of interest and files related to each.
species_dict = {
	"ath":"../data/pathways/plantcyc/aracyc_pathways.20180702", 
	"zma":"../data/pathways/plantcyc/corncyc_pathways.20180702", 
	"mtr":"../data/pathways/plantcyc/mtruncatulacyc_pathways.20180702", 
	"osa":"../data/pathways/plantcyc/oryzacyc_pathways.20180702", 
	"gmx":"../data/pathways/plantcyc/soycyc_pathways.20180702",
	"sly":"../data/pathways/plantcyc/tomatocyc_pathways.20180702"}




# Create objects related to pathways using either the KEGG or PlantCyc databases.
pathways = Pathways(species_dict, source="pmn")
pathways.describe()


# Look at histograms of pathway membership within each species.
for species in species_dict.keys():
	kegg_dict = pathways.get_pathway_dict(species=species, gene_dict=gene_dict)
	num_pathways_by_gene = [len(pathway_list) for pathway_list in kegg_dict.values()]
	histogram = np.histogram(num_pathways_by_gene, bins=np.arange(np.max(num_pathways_by_gene)), density=False)






# Repeat using information from KEGG instead.
pathways = Pathways(species_dict, source="kegg")
pathways.describe()

# Look at histograms of pathway membership within each species.
for species in species_dict.keys():
	kegg_dict = pathways.get_pathway_dict(species=species, gene_dict=gene_dict)
	num_pathways_by_gene = [len(pathway_list) for pathway_list in kegg_dict.values()]
	histogram = np.histogram(num_pathways_by_gene, bins=np.arange(np.max(num_pathways_by_gene)), density=False)

