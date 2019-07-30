import sys
import os
import pandas as pd


sys.path.append("../.")
import phenolog.nlp
import phenolog.ontology
import phenolog.similarity




# Read in a dataframe from a TAIR public release tsv containing fields of interest.
filename = "/Users/irbraun/phenolog/data/tair/Locus_Germplasm_Phenotype_20180702.txt"
usecols = ["LOCUS_NAME", "GERMPLASM_NAME", "PHENOTYPE", "PUBMED_ID"]
names = ["locus", "germplasm", "phenotype", "pubmed_id"]
renamed = {k:v for k,v in zip(usecols,names)}
df = pd.read_table(filename, usecols=usecols)
df.rename(columns=renamed, inplace=True)
df["id"] = [str(i) for i in df.index.values]


# Use a small subset of the data for this example.
df = df.head(100)



# Prepare a dictionary of phenotype descriptions where each has a unique ID value.
description_dict = {identifier:description for (identifier,description) in zip(df.id,df.phenotype)}
description_dict = {i:phenolog.nlp.get_clean_description(d) for (i,d) in description_dict.items()}



# Generate annotations and other structures needed for assessing similarity.
nc_jar = "/Users/irbraun/phenolog/lib/NobleCoder-1.0.jar"
merged_ontology_file = "../ontologies/mo.obo"
annotations_file = "../data/annotations/nc_annotations.tsv"
ontology_names = ["mo"]


annotations = phenolog.ontology.annotate_with_noble_coder(object_dict=description_dict, path_to_jarfile=nc_jar, ontology_names=ontology_names)
phenolog.ontology.write_annotations_to_tsv_file(annotations_dict=annotations, annotations_output_file=annotations_file)




# Generate dataframes for each method of assessing similarity between the descriptions.
print(phenolog.similarity.get_similarity_df_using_ontologies(merged_ontology_file, annotations_file, description_dict).head(30))