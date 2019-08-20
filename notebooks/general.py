import sys
import os
import pandas as pd
import numpy as np


sys.path.append("../.")
from phenolog.Dataset import Dataset
from phenolog.nlp import get_clean_description
from phenolog.ontology import get_term_dictionaries, annotate_with_rabin_karp, write_annotations_to_tsv_file
from phenolog.similarity import get_similarity_df_using_ontologies, get_similarity_df_using_doc2vec, get_similarity_df_using_bagofwords, get_similarity_df_using_setofwords
from phenolog.models import combine_dfs_with_name_dict, apply_mean, train_linear_regression_model, train_random_forest_model, apply_linear_regression_model, apply_random_forest_model




# Create a dataset of the phenotype descriptions and corresponding gene information.
dataset = Dataset()
dataset.add_data(pd.read_csv("../data/reshaped/arabidopsis_phenotypes.csv", lineterminator="\n"))
dataset.add_data(pd.read_csv("../data/reshaped/maize_phenotypes.csv", lineterminator="\n"))
dataset.add_data(pd.read_csv("../data/reshaped/ppn_phenotypes.csv", lineterminator="\n"))
dataset.add_data(pd.read_csv("../data/reshaped/ppn_phenes.csv", lineterminator="\n"))
dataset.describe()


# Subsample the data that is available.
dataset.randomly_subsample_dataset(n=10)
dataset.describe()



# Prepare a dictionary of phenotype descriptions where each has a unique ID value.
description_dict = dataset.get_description_dictionary()
description_dict = {i:get_clean_description(d) for (i,d) in description_dict.items()}



# Generate annotations and other structures needed for assessing similarity.
merged_ontology_file = "../ontologies/mo.obo"
annotations_file = "../data/annotations/annotations_with_mo.tsv"
doc2vec_model_file = "../gensim/apnews_dbow/doc2vec.bin"
term_dicts = get_term_dictionaries(ontology_obo_file=merged_ontology_file)
term_dict_fwd = term_dicts[0]
term_dict_rev = term_dicts[1]
annotations = annotate_with_rabin_karp(object_dict=description_dict, term_dict=term_dict_rev)
write_annotations_to_tsv_file(annotations_dict=annotations, annotations_output_file=annotations_file)




# Generate dataframes for each method of assessing similarity between the descriptions.
df1 = get_similarity_df_using_ontologies(merged_ontology_file, annotations_file, description_dict)
df2 = get_similarity_df_using_doc2vec(doc2vec_model_file, description_dict)
df3 = get_similarity_df_using_bagofwords(description_dict)
df4 = get_similarity_df_using_setofwords(description_dict)
dfs = [df1, df2, df3, df4]



# Create names for method used to generate similarity dataframes and map to them.
methods = ["ontology", "doc2vec", "bagofwords", "setofwords", "onto_unwt", "onto_wt"]
method_to_df = {k:v for (k,v) in zip(methods,dfs)}



# Merge the dataframes into a single dataframe.
merged_df = combine_dfs_with_name_dict(method_to_df)
print(merged_df)






# Combine the different graphs by taking the mean of each similarity value.
output_df = apply_mean(df=merged_df, predictor_columns=methods)
print(output_df)





# Combine the different graphs by training and applying a linear regression model.
merged_df["target_value"] = np.random.sample(merged_df.shape[0]) # Target values are floats between 0 and 1.
model = train_linear_regression_model(df=merged_df, predictor_columns=methods, target_column="target_value")
output_df = apply_linear_regression_model(df=merged_df, predictor_columns=methods, model=model)
print(output_df)




# Combine the different graphs by training and applying a random forest classifier.
merged_df["target_class"] = np.random.randint(0,2,merged_df.shape[0]) # Target classes are 0 or 1, randomly.
model = train_random_forest_model(df=merged_df, predictor_columns=methods, target_column="target_class")
output_df = apply_random_forest_model(df=merged_df, predictor_columns=methods, model=model)
print(output_df)







