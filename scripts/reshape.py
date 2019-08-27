import sys
import os
import pandas as pd
import numpy as np
import itertools
import re

sys.path.append("../.")
import phenolog.nlp.preprocess
import phenolog.utils.constants
from phenolog.nlp.preprocess import add_prefix








############### Data from the six different plant species from Oellrich, Walls et al. supplement ################



# Read in the phenotype descriptions dataframe from the Oellrich, Walls et al (2015) dataset.
filename = "../data/sources/cleaned/oellrich_walls_descriptions.csv"
usecols = ["species", "locus", "phenotype"]
usenames = ["species", "gene_names", "description"]
renamed = {k:v for k,v in zip(usecols,usenames)}
df = pd.read_csv(filename, usecols=usecols)
df.rename(columns=renamed, inplace=True)
df.drop_duplicates(keep="first", inplace=True)



df["terms"] = ""
df["evidence"] = ""
df["reference"] = ""



df = df[["species", "description", "gene_names", "pmid"]]
path = "../data/reshaped/pppn_phenotypes.csv"
df.to_csv(path, index=False)


# Read in the phene descriptions dataframe from the Oellrich, Walls et al (2015) dataset.
filename = "../data/phenotypes/ppn/oellrich_walls_descriptions.csv"
usecols = ["species", "locus", "phene"]
usenames = ["species", "gene_names", "description"]
renamed = {k:v for k,v in zip(usecols,usenames)}
df = pd.read_csv(filename, usecols=usecols)
df.rename(columns=renamed, inplace=True)
df.drop_duplicates(keep="first", inplace=True)
df["pmid"] = ""
df = df[["species", "description", "gene_names", "pmid"]]
path = "../data/reshaped/pppn_phenes.csv"
df.to_csv(path, index=False)











############### Data for Arabidopsis thaliana from TAIR database ################



# Create a dataset using the phenotype descriptions from the data release.
filename = "../data/sources/tair/Locus_Germplasm_Phenotype_20180702.txt"
usecols = ["LOCUS_NAME", "PHENOTYPE", "PUBMED_ID"]
usenames = ["gene_names", "description", "pmid"]
renamed = {k:v for k,v in zip(usecols,usenames)}
df = pd.read_table(filename, usecols=usecols)
df.rename(columns=renamed, inplace=True)
df["species"] = "ath"
df["pmid"] = ""
df = df[["species", "description", "gene_names", "pmid"]]
path = "../data/reshaped/arabidopsis_phenotypes.csv"
df.to_csv(path, index=False)






# Create a dataset using the ontology term annotations (PO and GO) from the data release.
filename = "../data/sources/tair/ATH_GO_GOSLIM.txt"
df_go = pd.read_table(filename, header=None, usecols=[0,3,4,5,9,12])
df_go.columns = ["locus","relationship","term_label","term_id","evidence_code","reference"]


filename = "../data/sources/tair/po_anatomy_gene_arabidopsis_tair.assoc"
df_po_spatial = pd.read_table(filename, header=None, skiprows=1, usecols=[2,4,5,6,9,10,11])
df_po_spatial.columns = ["symbol","term_id","references","evidence_code","name","synonyms","type"]

filename = "../data/sources/tair/po_temporal_gene_arabidopsis_tair.assoc"
df_po_temporal = pd.read_table(filename, header=None, skiprows=1, usecols=[2,4,5,6,9,10,11])
df_po_temporal.columns = ["symbol","term_id","references","evidence_code","name","synonyms","type"]














print(df.head())

sys.exit()



filename = "../data/sources/tair/po_temporal_gene_arabidopsis_tair.assoc"
df = pd.read_table(filename, header=None)
df = df[[0,3,4,5,9,12]]
df.columns = ["locus","relationship","term_label","term_id","evidence_code","reference"]
print(df.head())








sys.exit()











# Create a dataset using the 

























############### Data for maize from MaizeGDB database ################



# Read in the MaizeGDB dataset.
filename = "../data/phenotypes/maizegdb/pheno_genes.txt"
usecols = ["phenotype_name", "phenotype_description", "locus_name", "alleles", "locus_synonyms", "v3_gene_model", "v4_gene_model", "uniprot_id", "ncbi_gene"]
df = pd.read_table(filename, usecols=usecols)
df.fillna("", inplace=True)

# Column manipulation that's specific to this dataset.
df["description"] = np.vectorize(phenolog.nlp.preprocess.concatenate_descriptions)(df["phenotype_name"], df["phenotype_description"])
df["v3_gene_model"] = df["v3_gene_model"].apply(add_prefix, prefix=phenolog.utils.constants.REFGEN_V3_TAG)
df["v4_gene_model"] = df["v4_gene_model"].apply(add_prefix, prefix=phenolog.utils.constants.REFGEN_V4_TAG)
df["uniprot_id"] = df["uniprot_id"].apply(add_prefix, prefix=phenolog.utils.constants.UNIPROT_TAG)
df["ncbi_gene"] = df["ncbi_gene"].apply(add_prefix, prefix=phenolog.utils.constants.NCBI_TAG)
df["gene_names"] = np.vectorize(phenolog.nlp.preprocess.concatenate_with_bar_delim)(df["locus_name"], df["alleles"], df["locus_synonyms"], df["v3_gene_model"], df["v4_gene_model"], df["uniprot_id"], df["ncbi_gene"])
df["species"] = "zma"
df["pmid"] = ""
df = df[["species", "description", "gene_names", "pmid"]]
path = "../data/reshaped/maize_phenotypes.csv"
df.to_csv(path, index=False)












############### Data for rice from the OryzaBase database ################



def clean_oryzabase_symbol(string):
	string = phenolog.nlp.preprocess.remove_character(string, "*")
	names = phenolog.nlp.preprocess.handle_synonym_in_parentheses(string, min_length=4)
	names = [phenolog.nlp.preprocess.remove_enclosing_brackets(name) for name in names]
	names = phenolog.nlp.preprocess.remove_short_tokens(names, min_length=2)
	names_string = phenolog.nlp.preprocess.concatenate_with_bar_delim(*names)
	return(names_string)

def clean_oryzabase_symbol_synonyms(string):
	string = phenolog.nlp.preprocess.remove_character(string, "*")
	names = string.split(",")
	names = [name.strip() for name in names]
	names = [phenolog.nlp.preprocess.remove_enclosing_brackets(name) for name in names]
	names_string = phenolog.nlp.preprocess.concatenate_with_bar_delim(*names)
	return(names_string)

def clean_oryzabase_explainations(string):
	ontology_ids = get_list_of_ontology_ids(string)
	for ontology_id in ontology_ids:
		string = string.replace(ontology_id,"")
	string = phenolog.nlp.preprocess.remove_punctuation(string)
	return(string)


# 'Trait Class' is a value or values from a small specified hierarchy (https://shigen.nig.ac.jp/rice/oryzabase/traitclass/index/).
# 'Explanation' includes both free text, references in free text form, and ontology term IDs and labels.
# For a big chunk of the data, Explanation includes LOC_ locus names (like used in Oellrich, Walls et al instead), which are RAP ID or MUS ID.
# Account for those.
# How should the explaination column be parsed to retain only the pieces that are actually phenotype descriptions?



# Read in the Oryzabase dataset.
filename = "../data/phenotypes/oryzabase/OryzabaseGeneListEn_20190826010113.txt"
usecols = ["CGSNL Gene Symbol", "Gene symbol synonym(s)", "CGSNL Gene Name", "Gene name synonym(s)", "Protein Name", "Allele", "Explanation", "Trait Class", "Gene Ontology", "Trait Ontology", "Plant Ontology"]
usenames = ["CGSNL Gene Symbol", "Gene symbol synonym(s)", "CGSNL Gene Name", "Gene name synonym(s)", "Protein Name", "Allele", "Explanation", "description", "Gene Ontology", "Trait Ontology", "Plant Ontology"]
renamed = {k:v for k,v in zip(usecols,usenames)}
df = pd.read_table(filename, usecols=usecols, sep="\t")
df.rename(columns=renamed, inplace=True)
df.fillna("", inplace=True)

# Column manipulation that's specific to this dataset.
df["CGSNL Gene Symbol"] = df["CGSNL Gene Symbol"].apply(clean_oryzabase_symbol)
df["Gene symbol synonym(s)"] = df["Gene symbol synonym(s)"].apply(clean_oryzabase_symbol_synonyms)
df["CGSNL Gene Name"] = df["CGSNL Gene Name"].apply(lambda x: x.replace("_","").strip())
df["Gene name synonym(s)"] = df["Gene name synonym(s)"].apply(phenolog.nlp.preprocess.comma_delim_to_bar_delim)

df["Revised"] = df["Explanation"].apply(clean_oryzabase_explainations)
#df["RAP ID"] = 
#df["MUS ID"] = 
df["gene_names"] = np.vectorize(phenolog.nlp.preprocess.concatenate_with_bar_delim)(df["CGSNL Gene Symbol"], df["Gene symbol synonym(s)"], df["CGSNL Gene Name"], df["Gene name synonym(s)"])
df["species"] = "osa"
df["pmid"] = ""



for row in df.itertuples():
	print("\n")
	print("Ex", row.Explanation)
	print(get_list_of_ontology_ids(row.Explanation))
	print("GO", get_list_of_ontology_ids(row[9]))
	print("TO", get_list_of_ontology_ids(row[10]))
	print("PO", get_list_of_ontology_ids(row[11]))

df = df[["species", "description", "gene_names", "pmid", "Gene Ontology", "Explanation", "Revised"]]
df.to_csv("/Users/irbraun/Desktop/check.csv")














