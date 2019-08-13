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
dataset.describe()


# Subsample the data that is available.
dataset.randomly_subsample_dataset(n=5)
dataset.describe()



# Prepare a dictionary of phenotype descriptions where each has a unique ID value.
description_dict = dataset.get_description_dictionary()
description_dict = {i:phenolog.nlp.get_clean_description(d) for (i,d) in description_dict.items()}



# Generate annotations and other structures needed for assessing similarity.
merged_ontology_file = "../ontologies/mo.obo"
annotations_file = "../data/annotations/all_annotations.tsv"
doc2vec_model_file = "../gensim/apnews_dbow/doc2vec.bin"

term_dicts = phenolog.ontology.get_term_dictionaries(ontology_obo_file=merged_ontology_file)
term_dict_fwd = term_dicts[0]
term_dict_rev = term_dicts[1]
annotations = phenolog.ontology.annotate_with_rabin_karp(object_dict=description_dict, term_dict=term_dict_rev)
phenolog.ontology.write_annotations_to_tsv_file(annotations_dict=annotations, annotations_output_file=annotations_file)




# Generate dataframes for each method of assessing similarity between the descriptions.
df1 = phenolog.similarity.get_similarity_df_using_ontologies(merged_ontology_file, annotations_file, description_dict)
df2 = phenolog.similarity.get_similarity_df_using_doc2vec(doc2vec_model_file, description_dict)
df3 = phenolog.similarity.get_similarity_df_using_bagofwords(description_dict)
df4 = phenolog.similarity.get_similarity_df_using_setofwords(description_dict)

# Look at the contents of each dataframe.
print(df1)
print(df2)
print(df3)
print(df4)

# Create a mapping between names and each dataframe and corresponding weights.
dfs = [df1, df2, df3, df4]
df_names = ["ont","d2v","bow","sow"]
weights = [0.014, 0.914, 0.004, 0.006]
df_dict = {k:v for (k,v) in zip(df_names,dfs)}
wt_dict = {k:v for (k,v) in zip(df_names,weights)}

# Define which of the dataframes to use as the target values for fitting.
target_df = df4
reg_model = phenolog.combine.learn_weights_linear_regression(dfs_dict=df_dict, target_df=target_df)
df = phenolog.combine.combine_with_linear_model(df_dict, reg_model)
print(df)


