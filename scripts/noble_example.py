import sys
import os
import pandas as pd


sys.path.append("../.")
from phenolog.Dataset import Dataset
import phenolog.nlp
import phenolog.similarity
import phenolog.ontology
import phenolog.related
import phenolog.combine
import phenolog.utils




# Create a dataset of the phenotype descriptions and corresponding gene information.
dataset = Dataset()
dataset.add_data(pd.read_csv("../data/reshaped/arabidopsis_phenotypes.csv", lineterminator="\n"))
dataset.add_data(pd.read_csv("../data/reshaped/maize_phenotypes.csv", lineterminator="\n"))
dataset.add_data(pd.read_csv("../data/reshaped/ppn_phenotypes.csv", lineterminator="\n"))
dataset.add_data(pd.read_csv("../data/reshaped/ppn_phenes.csv", lineterminator="\n"))
dataset.randomly_subsample_dataset(n=100)


# Prepare a dictionary of phenotype descriptions where each has a unique ID value.
description_dict = dataset.get_description_dictionary()
description_dict = {i:phenolog.nlp.get_clean_description(d) for (i,d) in description_dict.items()}



# Generate annotations and other structures needed for assessing similarity.
nc_jar = "../lib/NobleCoder-1.0.jar"
merged_ontology_file = "../ontologies/mo.obo"
annotations_file = "../data/annotations/annotations_with_mo.tsv"
ontology_names = ["mo"]

annotations = phenolog.ontology.annotate_with_noble_coder(object_dict=description_dict, path_to_jarfile=nc_jar, ontology_names=ontology_names)
phenolog.ontology.write_annotations_to_tsv_file(annotations_dict=annotations, annotations_output_file=annotations_file)




# Generate dataframes for each method of assessing similarity between the descriptions.
print(phenolog.similarity.get_similarity_df_using_ontologies(merged_ontology_file, annotations_file, description_dict).head(30))