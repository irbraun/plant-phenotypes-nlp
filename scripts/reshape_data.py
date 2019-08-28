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
from phenolog.nlp.preprocess import concatenate_with_bar_delim
from phenolog.nlp.preprocess import get_ontology_ids
from phenolog.utils.utils import to_abbreviation








############### Data from the six different plant species from Oellrich, Walls et al. supplement ################



# Read in the phenotype and phene descriptions dataframe from the Oellrich, Walls et al (2015) dataset.
filename = "../data/sources/cleaned/oellrich_walls_dataset.csv"
usecols = ["Species", "gene symbol", "Gene Identifier", "allele (optional)", "gene name", "phenotype name", "phenotype description", 'atomized statement', 
	'primary entity1 ID', 'primary entity1 text', 'relation_to (optional)', 'primary entity2 ID (optional)', 'primary entity2 text (optional)', 
	'quality ID', 'quality text', 'PATO Qualifier ID (optional)', 'PATO Qualifier text (optional)',
	'secondary_entity1 ID (optional)', 'secondary_entity1 text (optional)', 'relation_to (optional)', 'secondary entity2 ID (optional)','secondary_entity2 text (opyional)',
	'developmental stage ID (optional)', 'developmental stage text (optional)', 'condition ID (optional)', 'condition text (optional)', 
	'Pubmed ID (optional)', 'Dominant, recessive, codominant, semi-dominant (optional)', 'Loss or gain of function (optional)', 'Comment on mode of inheritance (optional)']
df = pd.read_csv(filename, usecols=usecols)
df.fillna("", inplace=True)

# Create the columns that have the information we want to retain in the compressed format.
df["species"] = df["Species"].apply(to_abbreviation)
df["description"] = df["phenotype description"]
df["gene_names"] = np.vectorize(concatenate_with_bar_delim)(df["gene symbol"], df["Gene Identifier"], df["allele (optional)"], df["gene name"])
df["term_ids"] = np.vectorize(concatenate_with_bar_delim)(df["primary entity1 ID"], df["primary entity2 ID (optional)"], 
	df["quality ID"], df["PATO Qualifier ID (optional)"], df["secondary_entity1 ID (optional)"], df["secondary entity2 ID (optional)"], 
	df["developmental stage ID (optional)"], df["condition ID (optional)"])
df["pmid"] = df["Pubmed ID (optional)"]
df = df[["species", "gene_names", "description", "term_ids", "pmid"]]
df.drop_duplicates(keep="first", inplace=True)

# Save the resulting dataframe.
path = "../data/reshaped/pppn_dataset.csv"
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
df["term_ids"] = ""
df["pmid"] = ""
df = df[["species", "gene_names", "description", "term_ids", "pmid"]]
path = "../data/reshaped/arabidopsis_descriptions.csv"
df.to_csv(path, index=False)



# Create a dataset using the GO term annotations from the data release.
filename = "../data/sources/tair/ATH_GO_GOSLIM.txt"
df_go = pd.read_table(filename, header=None, usecols=[0,3,4,5,9,12])
df_go.columns = ["locus","relationship","term_label","term_id","evidence_code","reference"]
df_go["species"] = "ath"
df_go["gene_names"] = df_go["locus"]
df_go["description"] = ""
df_go["term_ids"] = df_go["term_id"]
df_go["pmid"] = ""
df_go = df_go[["species", "gene_names", "description", "term_ids", "pmid"]]
path = "../data/reshaped/arabidopsis_go_annotations.csv"
df_go.to_csv(path, index=False)



# Create a dataset using the PO term annotations from the data release.
filename = "../data/sources/tair/po_anatomy_gene_arabidopsis_tair.assoc"
df_po_spatial = pd.read_table(filename, header=None, skiprows=1, usecols=[2,4,5,6,9,10,11])
filename = "../data/sources/tair/po_temporal_gene_arabidopsis_tair.assoc"
df_po_temporal = pd.read_table(filename, header=None, skiprows=1, usecols=[2,4,5,6,9,10,11])
df_po = df_po_spatial.append(df_po_temporal, ignore_index=True)
df_po.columns= ["symbol","term_id","references","evidence_code","name","synonyms","type"]
df_po["species"] = "ath"
df_po["gene_names"] = np.vectorize(concatenate_with_bar_delim)(df_po["symbol"], df_po["synonyms"])
df_po["description"] = ""
df_po["term_ids"] = df_po["term_id"]
df_po["pmid"] = ""
df_po= df_po[["species", "gene_names", "description", "term_ids", "pmid"]]
path = "../data/reshaped/arabidopsis_po_annotations.csv"
df_po.to_csv(path, index=False)












############### Data for maize from MaizeGDB database ################



# Read in the MaizeGDB dataset.
filename = "../data/sources/maizegdb/pheno_genes.txt"
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
df["term_ids"] = ""
df["pmid"] = ""
df = df[["species", "gene_names", "description", "term_ids" ,"pmid"]]
path = "../data/reshaped/maize_descriptions.csv"
df.to_csv(path, index=False)










############### Data for rice from the OryzaBase database ################




# Small functions necessary for accurately parsing this file.
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
	ontology_ids = get_ontology_ids(string)
	for ontology_id in ontology_ids:
		string = string.replace(ontology_id,"")
	string = phenolog.nlp.preprocess.remove_punctuation(string)
	return(string)

# Description of some columns of interest in the data available at Oryzabase.
# 'Trait Class' is a value or values from a small specified hierarchy (https://shigen.nig.ac.jp/rice/oryzabase/traitclass/index/).
# 'Explanation' includes both free text, references in free text form, and ontology term IDs and labels.
# For a big chunk of the data, 'Explanation' includes LOC_[locus names] (like those used in Oellrich, Walls et al), which are 'RAP ID' or 'MUS ID'.
# TODO remove those aspects from the description field so that it's close to a phenotype description in atleast some cases.

# Read in the Oryzabase dataset.
filename = "../data/sources/oryzabase/OryzabaseGeneListEn_20190826010113.txt"
usecols = ["CGSNL Gene Symbol", "Gene symbol synonym(s)", "CGSNL Gene Name", "Gene name synonym(s)", "Protein Name", "Allele", 
	"Explanation", "Trait Class", "RAP ID", "MUS ID", "Gramene ID", "Gene Ontology", "Trait Ontology", "Plant Ontology"]
df = pd.read_table(filename, usecols=usecols, sep="\t")
df.fillna("", inplace=True)

df["CGSNL Gene Symbol"] = df["CGSNL Gene Symbol"].apply(clean_oryzabase_symbol)
df["Gene symbol synonym(s)"] = df["Gene symbol synonym(s)"].apply(clean_oryzabase_symbol_synonyms)
df["CGSNL Gene Name"] = df["CGSNL Gene Name"].apply(lambda x: x.replace("_","").strip())
df["Gene name synonym(s)"] = df["Gene name synonym(s)"].apply(phenolog.nlp.preprocess.comma_delim_to_bar_delim)
df["Gene Ontology"] = df["Gene Ontology"].apply(lambda x: concatenate_with_bar_delim(*get_ontology_ids(x))) 
df["Trait Ontology"] = df["Trait Ontology"].apply(lambda x: concatenate_with_bar_delim(*get_ontology_ids(x))) 
df["Plant Ontology"] = df["Plant Ontology"].apply(lambda x: concatenate_with_bar_delim(*get_ontology_ids(x))) 

df["species"] = "osa"
df["gene_names"] = np.vectorize(concatenate_with_bar_delim)(df["CGSNL Gene Symbol"], df["Gene symbol synonym(s)"], df["CGSNL Gene Name"], df["Gene name synonym(s)"], df["RAP ID"], df["MUS ID"])
df["description"] = df["Explanation"].apply(clean_oryzabase_explainations)
df["term_ids"] = np.vectorize(concatenate_with_bar_delim)(df["Gene Ontology"], df["Trait Ontology"], df["Plant Ontology"])
df["pmid"] = ""
df = df[["species", "gene_names", "description", "term_ids" ,"pmid"]]
path = "../data/reshaped/oryzabase_dataset.csv"
df.to_csv(path, index=False)












