import sys
import os
import pandas as pd
import numpy as np
import time
import multiprocessing as mp


from oats.utils.utils import function_wrapper
from oats.utils.utils import to_hms
from oats.utils.utils import save_to_pickle, load_from_pickle
from oats.datasets.dataset import Dataset
from oats.datasets.groupings import Groupings
from oats.nlp.preprocess import get_clean_description
from oats.annotation.ontology import Ontology
from oats.annotation.annotation import annotate_using_rabin_karp, annotate_using_noble_coder 
from oats.annotation.annotation import write_annotations_to_tsv_file, read_annotations_from_tsv_file


from oats.graphs.similarity import get_similarity_df_using_fastsemsim
from oats.graphs.similarity import get_similarity_df_using_doc2vec
from oats.graphs.similarity import get_similarity_df_using_bagofwords
from oats.graphs.similarity import get_similarity_df_using_setofwords
from oats.graphs.similarity import get_similarity_df_using_annotations_unweighted_jaccard
from oats.graphs.similarity import get_similarity_df_using_annotations_weighted_jaccard
from oats.graphs.data import combine_dfs_with_name_dict
from oats.graphs.data import subset_df_with_ids

from oats.graphs.models import apply_mean
from oats.graphs.models import train_linear_regression_model
from oats.graphs.models import apply_linear_regression_model
from oats.graphs.models import train_logistic_regression_model
from oats.graphs.models import apply_logistic_regression_model
from oats.graphs.models import train_random_forest_model
from oats.graphs.models import apply_random_forest_model

from oats.objectives.functions import classification
from oats.objectives.functions import consistency_index
from oats.graphs.graph import Graph
from oats.objectives.functions import pr




pd.set_option('mode.chained_assignment', None)




# Read in the text descriptions and associated genetic data.
dataset = Dataset()
dataset.add_data(pd.read_csv("../data/reshaped/arabidopsis_go_annotations.csv", lineterminator="\n"))
dataset.add_data(pd.read_csv("../data/reshaped/arabidopsis_descriptions.csv", lineterminator="\n"))
dataset.add_data(pd.read_csv("../data/reshaped/maize_descriptions.csv", lineterminator="\n"))
dataset.add_data(pd.read_csv("../data/reshaped/oryzabase_dataset.csv", lineterminator="\n"))
dataset.add_data(pd.read_csv("../data/reshaped/pppn_dataset.csv", lineterminator="\n"))

# Filtering the data that was available from those files.
dataset.collapse_by_first_gene_name()
dataset.filter_has_description()
dataset.filter_has_annotation()

# Randomly subsampling the data.
dataset.filter_random_k(k=20, seed=78263)






# Get dictionaries mapping IDs to text descriptions or genes.
start = time.perf_counter()
descriptions = dataset.get_description_dictionary()
descriptions = {i:get_clean_description(d) for (i,d) in descriptions.items()}
genes = dataset.get_gene_dictionary()
total= time.perf_counter()-start
print("\n\nTime for getting and cleaning descriptions and gene dict: {}\n\n".format(to_hms(total)))




# Create/read in the different objects for organizing gene groupings.
groupings_kegg = load_from_pickle(path="../data/pickles/kegg_pathways.pickle")
groupings_pmn = load_from_pickle(path="../data/pickles/pmn_pathways.pickle")
groupings_subset = load_from_pickle(path="../data/pickles/lloyd_subsets.pickle")
groupings_class = load_from_pickle(path="../data/pickles/lloyd_classes.pickle")


id_to_pathway_ids = groupings_kegg.get_forward_dict(genes)
pathway_id_to_ids = groupings_kegg.get_reverse_dict(genes)







# Setup some of the ontology and document embeddings stuff.
merged_ontology_file = "../ontologies/mo.obo"
annotations_file = "../data/annotations/mo_annotations.tsv"
doc2vec_model_file = "../gensim/enwiki_dbow/doc2vec.bin"
mo = Ontology(merged_ontology_file)

start = time.perf_counter()
annotations = annotate_using_rabin_karp(object_dict=descriptions, ontology=mo)
write_annotations_to_tsv_file(annotations_dict=annotations, annotations_output_path=annotations_file)
total= time.perf_counter()-start
print("\n\nTime for rabin karp and writing to file: {}\n\n".format(to_hms(total)))







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
start = time.perf_counter()
names = ["ontology", "doc2vec", "bagofwords", "setofwords", "onto_unwt", "onto_wt"]
name_to_df_mapping = {name:result[0] for (name,result) in zip(names,results)}
df = combine_dfs_with_name_dict(name_to_df_mapping)
total= time.perf_counter()-start
print("\n\nTime for combining the tables into one table: {}\n\n".format(to_hms(total)))



# Look at how long it took to build each pairwise similarity matrix.
print("\n\n")
print("Durations of generating each pairwise similarity matrix (hh:mm:ss)")
print("-----------------------------------------------------------------")
durations = [result[1] for result in results]
savings = total_time_mp/sum(durations)
for (name,duration) in zip(names,durations):
	print("{:15} {}".format(name, to_hms(duration)))
print("-----------------------------------------------------------------")
print("{:15} {}".format("total", to_hms(sum(durations))))
print("{:15} {} ({:.2%} of single thread time)".format("multiprocess", to_hms(total_time_mp), savings))
print("\n\n")





'''
# Load pathway object previously created and pickled.
pathways = load_from_pickle(path="../data/pickles/pmn_pathways.pickle")



# Find the subset of information (IDs) that have associated pathway data.
id_to_pathway_ids = pathways.get_fwd_pathway_dict(genes)
pathway_id_to_ids = pathways.get_rev_pathway_dict(genes)
ids_with_pathway_info = [identifer for (identifer,pathway_list) in id_to_pathway_ids.items() if len(pathway_list)>0]
df_train = subset_df_based_on_ids(df,ids_with_pathway_info)


# Assign target classes to each pair of IDs based on whether they share pathway membership.
target_classes = [int(len(set(id_to_pathway_ids[id1]).intersection(set(id_to_pathway_ids[id2])))>0) for (id1,id2) in zip(df_train["from"].values,df_train["to"].values)]
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
'''


start = time.perf_counter()
g = Graph(df=df, value="doc2vec")
total= time.perf_counter()-start
print("\n\nTime for creating the graph: {}\n\n".format(to_hms(total)))


start = time.perf_counter()
a,b = classification(graph=g, id_to_labels=id_to_pathway_ids, label_to_ids=pathway_id_to_ids)
mapp = consistency_index(graph=g, id_to_labels=id_to_pathway_ids, label_to_ids=pathway_id_to_ids)
for m in mapp.items():
	print(m)
total= time.perf_counter()-start
print("\n\nTime for getting the overall metrics: {}\n\n".format(to_hms(total)))








