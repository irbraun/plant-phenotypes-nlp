import sys
import os
import pandas as pd
import numpy as np
import itertools

sys.path.append("../.")
import phenolog.nlp
import phenolog.utils










# Read in the MaizeGDB dataset.
filename = "../data/maizegdb/pheno_genes.txt"
usecols = ["phenotype_name", "phenotype_description", "locus_name", "alleles", "locus_synonyms", "v3_gene_model", "v4_gene_model", "uniprot_id", "ncbi_gene"]
df = pd.read_table(filename, usecols=usecols)
df.fillna("", inplace=True)
df["description"] = np.vectorize(phenolog.nlp.concatenate_descriptions)(df["phenotype_name"], df["phenotype_description"])
df["v3_gene_model"] = df["v3_gene_model"].apply(phenolog.utils.add_tag, tag=phenolog.utils.refgen_v3_tag)
df["v4_gene_model"] = df["v4_gene_model"].apply(phenolog.utils.add_tag, tag=phenolog.utils.refgen_v4_tag)
df["uniprot_id"] = df["uniprot_id"].apply(phenolog.utils.add_tag, tag=phenolog.utils.uniprot_tag)
df["ncbi_gene"] = df["ncbi_gene"].apply(phenolog.utils.add_tag, tag=phenolog.utils.ncbi_tag)
df["gene_names"] = np.vectorize(phenolog.nlp.concatenate_with_bar_delim)(df["locus_name"], df["alleles"], df["locus_synonyms"], df["v3_gene_model"], df["v4_gene_model"], df["uniprot_id"], df["ncbi_gene"])
df["species"] = "zma"
df["pmid"] = ""
df = df[["species", "description", "gene_names", "pmid"]]
path = "../data/reshaped/maize_phenotypes.csv"
df.to_csv(path, index=False)







# Read in the phenotype descriptions dataframe from the Oellrich, Walls et al (2015) dataset.
filename = "../data/ppn/oellrich_walls_textdata_only.csv"
usecols = ["species", "locus", "phenotype"]
usenames = ["species", "gene_names", "description"]
renamed = {k:v for k,v in zip(usecols,usenames)}
df = pd.read_csv(filename, usecols=usecols)
df.rename(columns=renamed, inplace=True)
df.drop_duplicates(keep="first", inplace=True)
df["pmid"] = ""
df = df[["species", "description", "gene_names", "pmid"]]
path = "../data/reshaped/ppn_phenotypes.csv"
df.to_csv(path, index=False)






# Read in the phene descriptions dataframe from the Oellrich, Walls et al (2015) dataset.
filename = "/Users/irbraun/phenolog/data/ppn/oellrich_walls_textdata_only.csv"
usecols = ["species", "locus", "phene"]
usenames = ["species", "gene_names", "description"]
renamed = {k:v for k,v in zip(usecols,usenames)}
df = pd.read_csv(filename, usecols=usecols)
df.rename(columns=renamed, inplace=True)
df.drop_duplicates(keep="first", inplace=True)
df["pmid"] = ""
df = df[["species", "description", "gene_names", "pmid"]]
path = "../data/reshaped/ppn_phenes.csv"
df.to_csv(path, index=False)







# Read in a dataframe from a TAIR public release tsv containing fields of interest.
filename = "/Users/irbraun/phenolog/data/tair/Locus_Germplasm_Phenotype_20180702.txt"
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





