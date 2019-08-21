import sys
import os
import pandas as pd
import numpy as np


sys.path.append("../.")

from phenolog.datasets.dataset import Dataset
from phenolog.nlp.preprocess import get_clean_description
from phenolog.annotation.ontology import Ontology
from phenolog.annotation.annotation import annotate_using_rabin_karp, annotate_using_noble_coder 
from phenolog.annotation.annotation import write_annotations_to_tsv_file, read_annotations_from_tsv_file

from phenolog.graphs.similarity import get_similarity_df_using_fastsemsim
from phenolog.graphs.similarity import get_similarity_df_using_doc2vec
from phenolog.graphs.similarity import get_similarity_df_using_bagofwords
from phenolog.graphs.similarity import get_similarity_df_using_setofwords
from phenolog.graphs.similarity import get_similarity_df_using_annotations_unweighted_jaccard
from phenolog.graphs.similarity import get_similarity_df_using_annotations_weighted_jaccard

from phenolog.graphs.models import combine_dfs_with_name_dict
from phenolog.graphs.models import subset_based_on_ids
from phenolog.graphs.models import apply_mean
from phenolog.graphs.models import train_linear_regression_model
from phenolog.graphs.models import apply_linear_regression_model
from phenolog.graphs.models import train_random_forest_model
from phenolog.graphs.models import apply_random_forest_model

from phenolog.datasets.pathways import Pathways




# Read in the text descriptions and associated genetic data.
dataset = Dataset()
dataset.add_data(pd.read_csv("../data/reshaped/arabidopsis_phenotypes.csv", lineterminator="\n"))
dataset.add_data(pd.read_csv("../data/reshaped/maize_phenotypes.csv", lineterminator="\n"))
dataset.add_data(pd.read_csv("../data/reshaped/ppn_phenotypes.csv", lineterminator="\n"))
dataset.add_data(pd.read_csv("../data/reshaped/ppn_phenes.csv", lineterminator="\n"))
dataset.randomly_subsample_dataset(n=50, seed=78263)



# Get dictionaries mapping IDs to text descriptions or genes.
descriptions = dataset.get_description_dictionary()
descriptions = {i:get_clean_description(d) for (i,d) in descriptions.items()}
genes = dataset.get_gene_dictionary()



# Do some ontology and embedding related stuff.
merged_ontology_file = "../ontologies/mo.obo"
annotations_file = "../data/annotations/annotations_with_mo.tsv"
doc2vec_model_file = "../gensim/apnews_dbow/doc2vec.bin"
mo = Ontology(merged_ontology_file)
annotations = annotate_using_rabin_karp(descriptions, mo)
write_annotations_to_tsv_file(annotations, annotations_file)



# Generate the similarity matrices.
#df1 = get_similarity_df_using_fastsemsim(merged_ontology_file, annotations_file, descriptions)
#df2 = get_similarity_df_using_doc2vec(doc2vec_model_file, descriptions)
df3 = get_similarity_df_using_bagofwords(descriptions)
df4 = get_similarity_df_using_setofwords(descriptions)
df5 = get_similarity_df_using_annotations_unweighted_jaccard(annotations, mo)
#df6 = get_similarity_df_using_annotations_weighted_jaccard(annotations, mo)
dfs = [df3, df4, df5]

for df in dfs:
	print(df)


methods = ["ontology", "doc2vec", "bagofwords", "setofwords", "onto_unwt", "onto_wt"]
method_to_df = {k:v for (k,v) in zip(methods,dfs)}
merged_df = combine_dfs_with_name_dict(method_to_df)
print(merged_df.head())



# Create the pathways object.
species_dict = {
    "ath":"../data/pathways/plantcyc/aracyc_pathways.20180702", 
    "zma":"../data/pathways/plantcyc/corncyc_pathways.20180702", 
    "mtr":"../data/pathways/plantcyc/mtruncatulacyc_pathways.20180702", 
    "osa":"../data/pathways/plantcyc/oryzacyc_pathways.20180702", 
    "gmx":"../data/pathways/plantcyc/soycyc_pathways.20180702",
    "sly":"../data/pathways/plantcyc/tomatocyc_pathways.20180702"}
pathways = Pathways(species_dict, source="pmn")







# How to generate target values using the pathways object?
# Need to get subsets of the dfs which only deal with the information we have knowledge for in pathways.
pathway_membership = pathways.get_pathway_dict(genes)
ids_with_pathway_info = [identifer for (identifer,pathway_list) in pathway_membership.items() if len(pathway_list)>0]
print(ids_with_pathway_info)
sub_dfs = [subset_based_on_ids(df, ids_with_pathway_info) for df in dfs]



for df in sub_dfs:
	print(df)


for df in sub_dfs:
	membership_array = [int(len(set(pathway_membership[id1]).intersection(set(pathway_membership[id2])))>0) for (id1,id2) in zip(df["from"].values,df["to"].values)]
	df["class"] = membership_array 
	print(df)








# Want to learn a model on just this subset of datathat is known.






















"""
Read in the data and get a dataset.

Do all the annotations to produce each individual matrix.

Save all those matrices.

Learn a random forest model for combining those.
Produce an additional matrix from everything.
That one would be overfitted if using the same data to produce it?


Have another script that actually compares those matrices
given a membership dictionary of which groups those belong to.
Load in all the Lloyd Meinke data so that that's in there.











"""













'''

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

'''





