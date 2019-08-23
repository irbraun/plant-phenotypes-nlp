import sys
import os
import pandas as pd
import numpy as np
import time
import multiprocessing as mp

sys.path.append("../.")

from phenolog.utils.utils import function_wrapper
from phenolog.utils.utils import to_hms
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

from phenolog.graphs.data import combine_dfs_with_name_dict
from phenolog.graphs.data import subset_df_based_on_ids
from phenolog.graphs.models import apply_mean
from phenolog.graphs.models import train_linear_regression_model
from phenolog.graphs.models import apply_linear_regression_model
from phenolog.graphs.models import train_logistic_regression_model
from phenolog.graphs.models import apply_logistic_regression_model
from phenolog.graphs.models import train_random_forest_model
from phenolog.graphs.models import apply_random_forest_model

from phenolog.datasets.pathways import Pathways



pd.set_option('mode.chained_assignment', None)




# Read in the text descriptions and associated genetic data.
dataset = Dataset()
dataset.add_data(pd.read_csv("../data/reshaped/arabidopsis_phenotypes.csv", lineterminator="\n"))
dataset.add_data(pd.read_csv("../data/reshaped/maize_phenotypes.csv", lineterminator="\n"))
dataset.add_data(pd.read_csv("../data/reshaped/ppn_phenotypes.csv", lineterminator="\n"))
dataset.add_data(pd.read_csv("../data/reshaped/ppn_phenes.csv", lineterminator="\n"))
dataset.randomly_subsample_dataset(n=40, seed=78263)

# Get dictionaries mapping IDs to text descriptions or genes.
descriptions = dataset.get_description_dictionary()
descriptions = {i:get_clean_description(d) for (i,d) in descriptions.items()}
genes = dataset.get_gene_dictionary()

# Setup some of the ontology and document embeddings stuff.
merged_ontology_file = "../ontologies/mo.obo"
annotations_file = "../data/annotations/mo_annotations.tsv"
doc2vec_model_file = "../gensim/enwiki_dbow/doc2vec.bin"
mo = Ontology(merged_ontology_file)
annotations = annotate_using_rabin_karp(object_dict=descriptions, ontology=mo)
write_annotations_to_tsv_file(annotations_dict=annotations, annotations_output_path=annotations_file)







# Setup for creating the pairwise similarity matrices using seperate cores.
# The key is the function to use as a variable, and the values are the arguments to unpack as a list.
functions_and_args = {
	get_similarity_df_using_fastsemsim:[merged_ontology_file, annotations_file, descriptions, True],
	get_similarity_df_using_doc2vec:[doc2vec_model_file, descriptions, True],
	get_similarity_df_using_bagofwords:[descriptions, True],
	get_similarity_df_using_setofwords:[descriptions, True],
	get_similarity_df_using_annotations_unweighted_jaccard:[annotations, mo, True],
	get_similarity_df_using_annotations_weighted_jaccard:[annotations, mo, True]}

# Use parallel processing the build all the similarity matrices.
start_time_mp = time.perf_counter()
pool = mp.Pool(mp.cpu_count())
results = [pool.apply_async(function_wrapper, args=(function, args)) for (function, args) in functions_and_args.items()]
results = [result.get() for result in results]
pool.close()
pool.join()    
total_time_mp = time.perf_counter()-start_time_mp




# Create a mapping between method names and the similarity matrices that were generated.
names = ["ontology", "doc2vec", "bagofwords", "setofwords", "onto_unwt", "onto_wt"]
name_to_df_mapping = {name:result[0] for (name,result) in zip(names,results)}
df = combine_dfs_with_name_dict(name_to_df_mapping)


# Look at how long it took to build each pairwise similarity matrix.
print("\n\n")
print("Durations of generating each pairwise similarity matrix (seconds)")
print("-----------------------------------------------------------------")
durations = [result[1] for result in results]
savings = total_time_mp/sum(durations)
for (name,duration) in zip(names,durations):
	print("{:15} {}".format(name, to_hms(duration)))
print("-----------------------------------------------------------------")
print("{:15} {}".format("total", to_hms(sum(durations))))
print("{:15} {} ({:.2%} of single thread time)".format("multiprocess", to_hms(total_time_mp), savings))
print("\n\n")






# Create the pathways object.
species_dict = {
    "ath":"../data/pathways/plantcyc/aracyc_pathways.20180702", 
    "zma":"../data/pathways/plantcyc/corncyc_pathways.20180702", 
    "mtr":"../data/pathways/plantcyc/mtruncatulacyc_pathways.20180702", 
    "osa":"../data/pathways/plantcyc/oryzacyc_pathways.20180702", 
    "gmx":"../data/pathways/plantcyc/soycyc_pathways.20180702",
    "sly":"../data/pathways/plantcyc/tomatocyc_pathways.20180702"}
pathways = Pathways(species_dict, source="pmn")



# Find the subset of information (IDs) that have associated pathway data.
pathway_membership = pathways.get_pathway_dict(genes)
ids_with_pathway_info = [identifer for (identifer,pathway_list) in pathway_membership.items() if len(pathway_list)>0]
df_train = subset_df_based_on_ids(df,ids_with_pathway_info)


# Assign target classes to each pair of IDs based on whether they share pathway membership.
target_classes = [int(len(set(pathway_membership[id1]).intersection(set(pathway_membership[id2])))>0) for (id1,id2) in zip(df_train["from"].values,df_train["to"].values)]
df_train.loc[:,"class"] = target_classes





# Train the random forest on this subset, then apply to the whole dataset.
model = train_random_forest_model(df=df_train, predictor_columns=names, target_column="class")
df_rf = apply_random_forest_model(df=df, predictor_columns=names, model=model)
df.loc[:,"ranforest"] = df_rf["similarity"].values
print(df)




# Train a logistic regression model on this subset, then apply to the whole dataset.
model = train_logistic_regression_model(df=df_train, predictor_columns=names, target_column="class")
df_lr = apply_logistic_regression_model(df=df, predictor_columns=names, model=model)
df.loc[:,"logreg"] = df_lr["similarity"].values
print(df)












