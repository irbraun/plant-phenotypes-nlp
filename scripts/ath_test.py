import sys
import os
import pandas as pd
import numpy as np
import time
import multiprocessing as mp

sys.path.append("../.")

from phenolog.utils.utils import function_wrapper
from phenolog.utils.utils import to_hms
from phenolog.utils.utils import save_to_pickle, load_from_pickle
from phenolog.datasets.dataset import Dataset
from phenolog.datasets.groupings import Groupings
from phenolog.nlp.preprocess import get_clean_description
from phenolog.annotation.ontology import Ontology
from phenolog.annotation.annotation import annotate_using_rabin_karp, annotate_using_noble_coder 
from phenolog.annotation.annotation import write_annotations_to_tsv_file, read_annotations_from_tsv_file





pd.set_option('mode.chained_assignment', None)





# Loading datasets of text descriptions, annotations, and groupings





# Read in the text descriptions and associated genetic data.
dataset = Dataset()
dataset.add_data(pd.read_csv("../data/reshaped/arabidopsis_go_annotations.csv", lineterminator="\n"))
dataset.add_data(pd.read_csv("../data/reshaped/arabidopsis_po_annotations.csv", lineterminator="\n"))
dataset.add_data(pd.read_csv("../data/reshaped/arabidopsis_descriptions.csv", lineterminator="\n"))
dataset.add_data(pd.read_csv("../data/reshaped/pppn_dataset.csv", lineterminator="\n"))

# Filtering the data that was available from those files.
dataset.collapse_by_first_gene_name()
dataset.filter_has_annotation()
dataset.filter_has_description()

# Get dictionaries mapping IDs to text descriptions or genes.
descriptions = dataset.get_description_dictionary()
descriptions = {i:get_clean_description(d) for (i,d) in descriptions.items()}
genes = dataset.get_gene_dictionary()





# Create/read in the different objects for organizing gene groupings.
groupings_kegg = load_from_pickle(path="../data/pickles/kegg_pathways.pickle")
groupings_pmn = load_from_pickle(path="../data/pickles/pmn_pathways.pickle")
groupings_subset = load_from_pickle(path="../data/pickles/lloyd_subsets.pickle")
groupings_class = load_from_pickle(path="../data/pickles/lloyd_classes.pickle")
groupings_list = [groupings_kegg, groupings_pmn, groupings_subset]


# goal is to use one of these for testing.
# and then reserve the others ones for training data that can be used to train an ensemble learner.
# throw out any of the data from that second portion that is overlapping with what's in the first part.
# (even though it's not from the same source, it might be the same information).

# then the two different pairwise similarity matrices can be trained separately, no reason
# to generate the connections between them for the purposes of evaluation, this should save time.

# does this mean it would be helpful to have a method for adding new classifications to the groupign methods?
# that means, would have to:
# 1) get rid of or standardize columns in the dfs part of it.
# 2) adding a check when adding to warn if using a string (as pathway/group ID) that is already present in there.
# 3) anything else?

# As an alternative, could just add a static method there that allows for merging multiple dataset objects.
# all that's needed is a way to get an updates forward and reverse dictionary that has everything in there.




id_to_group_ids = groupings_subset.get_forward_dict(genes)
group_id_to_ids = groupings_subset.get_reverse_dict(genes)


# Retain only the datapoints that we have group information for.
ids_with_group_mapping = [identifier for (identifier,groups_list) in id_to_group_ids.items() if not len(groups_list)==0]
dataset.filter_with_ids(ids_with_group_mapping)
dataset.filter_random_k(k=200, seed=98372)
dataset.describe()







# Have to retrieve all the dictionaries again because the dataset has been subsampled.
descriptions = dataset.get_description_dictionary()
descriptions = {i:get_clean_description(d) for (i,d) in descriptions.items()}
genes = dataset.get_gene_dictionary()
id_to_group_ids = groupings_subset.get_forward_dict(genes)
group_id_to_ids = groupings_subset.get_reverse_dict(genes)











from phenolog.graphs.similarity import get_similarity_df_using_fastsemsim
from phenolog.graphs.similarity import get_similarity_df_using_doc2vec
from phenolog.graphs.similarity import get_similarity_df_using_bagofwords
from phenolog.graphs.similarity import get_similarity_df_using_setofwords
from phenolog.graphs.similarity import get_similarity_df_using_annotations_unweighted_jaccard
from phenolog.graphs.similarity import get_similarity_df_using_annotations_weighted_jaccard
from phenolog.graphs.data import combine_dfs_with_name_dict
from phenolog.graphs.data import subset_df_with_ids




# Finding pairwise similarities between the different representations


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











from phenolog.graphs.models import apply_mean
from phenolog.graphs.models import train_linear_regression_model
from phenolog.graphs.models import apply_linear_regression_model
from phenolog.graphs.models import train_logistic_regression_model
from phenolog.graphs.models import apply_logistic_regression_model
from phenolog.graphs.models import train_random_forest_model
from phenolog.graphs.models import apply_random_forest_model













''' ML part that adds additional colums to the dataframe by splitting off training data, learning model, applying to whole.

# Find the subset of information (IDs) in the dataset that have associated grouping data.
ids_with_group_mapping = [identifier for (identifier,groups_list) in id_to_group_ids.items() if not len(groups_list)==0]
df_train = subset_df_with_ids(df, ids_with_group_mapping)

# Create target class (1 or 0) column in the dataframe to be used for training.
target_classes = [int(len(set(id_to_group_ids[id1]).intersection(set(id_to_group_ids[id2])))>0) for (id1,id2) in zip(df_train["from"].values,df_train["to"].values)]
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








# Evaluation Step
from phenolog.objectives.functions import classification
from phenolog.objectives.functions import consistency_index
from phenolog.graphs.graph import Graph
from phenolog.objectives.functions import pr



graphs = [Graph(df=df, value=name) for name in names]
results = [classification(graph=graph, id_to_labels=id_to_group_ids, label_to_ids=group_id_to_ids) for graph in graphs]
cimaps = {name:consistency_index(graph=graph, id_to_labels=id_to_group_ids, label_to_ids=group_id_to_ids) for (name,graph) in zip(names,graphs)}


a = pd.DataFrame(cimaps)
print(a)








