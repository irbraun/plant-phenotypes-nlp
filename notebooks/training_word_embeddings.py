#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import gensim
import random
import sys
import glob
import os
import datetime
from nltk import sent_tokenize
from nltk import word_tokenize
from scipy.spatial.distance import cosine
import warnings
from gensim.parsing.preprocessing import strip_non_alphanum, stem_text, preprocess_string
from gensim.utils import simple_preprocess
from gensim.models.callbacks import CallbackAny2Vec
from itertools import product
from sklearn.metrics import precision_recall_curve

warnings.simplefilter('ignore')

sys.path.append("../../oats")
from oats.annotation.ontology import Ontology
from oats.distances import pairwise as pw
from oats.utils.utils import flatten


# ### 1. Creating datasets of sentences to traing word embedding models

# In[2]:


# Input paths to text datasets.
plant_abstracts_corpus_path = "../data/corpus_related_files/untagged_text_corpora/phenotypes_all.txt"
plant_phenotype_descriptions_path = "../../plant-data/genes_texts_annots.csv"


# In[ ]:


# Preparing the dataset that combines the dataset of plant phenotype descriptions and scrapped abstracts.
corpus = open(plant_abstracts_corpus_path, 'r').read()
sentences_from_corpus = sent_tokenize(corpus)
phenotype_descriptions = " ".join(pd.read_csv(plant_phenotype_descriptions_path)["descriptions"].values)
times_to_duplicate_phenotype_dataset = 5
sentences_from_descriptions = sent_tokenize(phenotype_descriptions)
sentences_from_descriptions_duplicated = list(np.repeat(sentences_from_descriptions, times_to_duplicate_phenotype_dataset))
sentences_from_corpus_and_descriptions = sentences_from_corpus+sentences_from_descriptions_duplicated
random.shuffle(sentences_from_corpus_and_descriptions)
random.shuffle(sentences_from_corpus)
random.shuffle(sentences_from_descriptions)
sentences_from_corpus_and_descriptions = [preprocess_string(s) for s in sentences_from_corpus_and_descriptions]
sentences_from_corpus = [preprocess_string(s) for s in sentences_from_corpus]
sentences_from_descriptions = [preprocess_string(s) for s in sentences_from_descriptions]
assert len(sentences_from_corpus_and_descriptions) == len(sentences_from_corpus)+(times_to_duplicate_phenotype_dataset*len(sentences_from_descriptions))
print(len(sentences_from_corpus_and_descriptions))
print(len(sentences_from_corpus))
print(len(sentences_from_descriptions))


# ### 2. Training and saving models with hyperparameter grid search

# In[ ]:


# Defining grid search parameters for training word embedding models.
training_sentence_sets =  [(sentences_from_corpus_and_descriptions,"both"), (sentences_from_corpus,"abstracts"), (sentences_from_descriptions,"dataset")]
training_sentence_sets =  [(sentences_from_corpus_and_descriptions,"both")]
dimensions = [(x,"dim{}".format(str(x))) for x in [50, 100, 150, 200]]
num_epochs = [(500,"500")]
min_alpha = [(0.0001,"a")]
alpha = [(0.025,"s")]
min_count = [(x,"min{}".format(str(x))) for x in [3,5]]
window = [(x,"window{}".format(str(x))) for x in [5,8]]
hyperparameter_sets = list(product(
    training_sentence_sets, 
    dimensions, 
    num_epochs, 
    min_alpha, 
    alpha, 
    min_count, 
    window))
print(len(hyperparameter_sets))


# In[ ]:


class LossLogger(CallbackAny2Vec):
    def __init__(self):
        self.epochs = []
        self.epoch = 1
        self.losses = []
        self.deltas = []
    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 1:
            delta = loss
        else:
            delta = loss- self.loss_previous_step
        self.loss_previous_step=loss
        self.losses.append(loss)
        self.epochs.append(self.epoch)
        self.epoch += 1
        self.deltas.append(delta)


# In[ ]:


def train_and_save_word_embedding_model(output_dir, hyperparameters):
    
    # Producing the path to the output file named according to hyperparameters.
    model_filename = "word2vec_{}.model".format("_".join([x[1] for x in hyperparameters]))
    output_path = os.path.join(output_dir, model_filename)
    
    # Get the hyperparameter values.
    sents, dim, epochs, min_a, a, min_count, window = [x[0] for x in hyperparameters]                      
    
    # Training the word2vec neural network with the current set of hyperparameters. 
    model = gensim.models.Word2Vec(sg=1, min_count=min_count, window=window, size=dim, workers=4, alpha=a, min_alpha=min_a)
    model.build_vocab(sents)
    loss_logger = LossLogger()
    model.train(sents, epochs=epochs, total_examples=model.corpus_count, compute_loss=True, callbacks=[loss_logger])
               
    # Saving the model to a file.
    model.save(output_path)
    print("saving {}".format(model_filename))


# In[ ]:


# Calling the model creation function iteratively through the hyperparameter grid search.
output_models_directory = "../models/plants_sg"
for h in hyperparameter_sets:
    train_and_save_word_embedding_model(output_models_directory, h)


# ### 3. Using ontologies to generate datasets of closely related domain concepts

# In[49]:


# Add in the section here for creating the small validation set.
# This creates a dataframe that has all the parent child term pairs in it including synonyms, and excluding
# the term pairs that have an explicit overlap in a token because that's not useful for embedding validation.

# Load the ontology and term information.
path = "../ontologies/pato.obo"
ont = Ontology(path)
term_ids_and_names = [(t.id,t.name) for t in ont.terms() if "obsolete" not in t.name]

# Also including all the synonym information here.
term_ids_and_names_with_synonyms = []
for i,name in term_ids_and_names:
    term_ids_and_names_with_synonyms.append((i," ".join(ont.term_to_tokens[i])))

key_to_annotations = {i:[x[0]] for i,x in enumerate(term_ids_and_names)}
key_to_term_id = {i:x[0] for i,x in enumerate(term_ids_and_names)}
key_to_text_string = {i:x[1] for i,x in enumerate(term_ids_and_names_with_synonyms)}
key_to_preprocessed_text_string = {i:" ".join(preprocess_string(s)) for i,s in key_to_text_string.items()}

# Get mappings that define which terms are very close to which others ones in the ontology structure.
parents = {}
children = {}
for term in ont.terms():
    parents[term.id] = [t.id for t in term.superclasses(with_self=False, distance=1)]
    children[term.id] = [t.id for t in term.subclasses(with_self=False, distance=1)]
siblings = {}
for term in ont.terms():
    siblings[term.id] = flatten([[t for t in children[parent_id] if t!=term.id] for parent_id in parents[term.id]])
assert len(parents) == len(children)
assert len(parents) == len(siblings)
any_close = {}
for key in parents.keys():
    any_close[key] = flatten([parents[key],children[key],siblings[key]])
    any_close[key] = flatten([parents[key],children[key]])

df = pw.with_annotations(key_to_annotations, ont, "jaccard", tfidf=False).edgelist
df = df[df["from"]!=df["to"]]
df["from_id"] = df["from"].map(lambda x: key_to_term_id[x])
df["to_id"] = df["to"].map(lambda x: key_to_term_id[x])
df["from_text"] = df["from"].map(lambda x: key_to_text_string[x])
df["to_text"] = df["to"].map(lambda x: key_to_text_string[x])
df["close"] = df.apply(lambda x: x["to_id"] in any_close[x["from_id"]], axis=1)
df["token_overlap"] = df.apply(lambda x: len(set(x["from_text"].split()).intersection(set(x["to_text"].split())))>0, axis=1)
df = df[(df["token_overlap"]==False) & (df["close"]==True)]
df.head(20)


# In[50]:


df.sample(10)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


def build_validation_df_from_ontology(path):
    
    # Load the ontology and term information.
    ont = Ontology(path)
    term_ids_and_names = [(t.id,t.name) for t in ont.terms() if "obsolete" not in t.name]
    key_to_annotations = {i:[x[0]] for i,x in enumerate(term_ids_and_names)}
    key_to_term_id = {i:x[0] for i,x in enumerate(term_ids_and_names)}
    key_to_text_string = {i:x[1] for i,x in enumerate(term_ids_and_names)}
    key_to_preprocessed_text_string = {i:" ".join(preprocess_string(s)) for i,s in key_to_text_string.items()}
    
    # Get mappings that define which terms are very close to which others ones in the ontology structure.
    parents = {}
    children = {}
    for term in ont.terms():
        parents[term.id] = [t.id for t in term.superclasses(with_self=False, distance=1)]
        children[term.id] = [t.id for t in term.subclasses(with_self=False, distance=1)]
    siblings = {}
    for term in ont.terms():
        siblings[term.id] = flatten([[t for t in children[parent_id] if t!=term.id] for parent_id in parents[term.id]])
    assert len(parents) == len(children)
    assert len(parents) == len(siblings)
    any_close = {}
    for key in parents.keys():
        any_close[key] = flatten([parents[key],children[key],siblings[key]])
        
        
    df = pw.with_annotations(key_to_annotations, ont, "jaccard", tfidf=False).edgelist
    df = df[df["from"]!=df["to"]]
    df["from_id"] = df["from"].map(lambda x: key_to_term_id[x])
    df["to_id"] = df["to"].map(lambda x: key_to_term_id[x])
    df["from_text"] = df["from"].map(lambda x: key_to_text_string[x])
    df["to_text"] = df["to"].map(lambda x: key_to_text_string[x])
    df["close"] = df.apply(lambda x: x["to_id"] in any_close[x["from_id"]], axis=1)
    df["token_overlap"] = df.apply(lambda x: len(set(x["from_text"].split()).intersection(set(x["to_text"].split())))>0, axis=1)
    df.head(20)
    
    positive_df = df[(df["token_overlap"]==False) & (df["close"]==True)]
    negative_df = df[(df["token_overlap"]==False) & (df["close"]==False)]
    assert negative_df.shape[0]+positive_df.shape[0] == df[df["token_overlap"]==False].shape[0]
    num_positive_examples = positive_df.shape[0]
    validation_df = pd.concat([positive_df, negative_df.sample(num_positive_examples, random_state=2)])
    del df
    return(validation_df, key_to_preprocessed_text_string)


# In[ ]:


pato_validation_df, pato_key_to_preprocessed_text_string = build_validation_df_from_ontology("../ontologies/pato.obo")
po_validation_df, po_key_to_preprocessed_text_string = build_validation_df_from_ontology("../ontologies/po.obo")
print("done")


# In[ ]:


test_words = ["auxin","leaves","dwarfism","roots","tip","anatomy","abnormal","hair","late","flowering"]
test_words = [preprocess_string(word)[0] for word in test_words]
test_words


# ### 3.5 Validation dataset of paraphrased phenotype sentences

# In[ ]:


from pathlib import Path
paraphrase_output_path = "/Users/irbraun/phenologs-with-oats/data/corpus_related_files/paraphrasing/from_corpus.txt"
f = Path(paraphrase_output_path)
if not f.exists():
    corpus = open(plant_abstracts_corpus_path, 'r').read()
    sentences_from_corpus = sent_tokenize(corpus)
    random.shuffle(sentences_from_corpus)
    sentences_from_corpus = [s for s in sentences_from_corpus if len(s.split())<=20]
    with open(paraphrase_output_path, "w") as f:
        for s in sentences_from_corpus[:100]:
            f.write(s+"\n")


# ### 4. Evaluating word embedding models on the validation dataset of related concepts

# In[ ]:


models_dir = "../models/plants_sg"
output_path_for_results = "../models/plants_sg/{}_validation.csv".format(datetime.datetime.now().strftime('%m_%d_%Y_h%Hm%Ms%S'))
rows = []
header = ["model","pato","po"]
header.extend(["word"]*len(test_words))
for path in glob.glob(os.path.join(models_dir,"*.model")):
    
    print("validating model at {}".format(path))
    model = gensim.models.Word2Vec.load(path)
    model_name = os.path.basename(path)
    
    
    if model_name == "word2vec.model":
        continue
    
    
    
    row_items = []
    row_items.append(model_name)
    
    # Use the pairwise interface to actually get a set of document vectors and calculate maximum F_1 values.
    validation_df = pato_validation_df
    result = pw.with_word2vec(model, pato_key_to_preprocessed_text_string, "cosine", "mean")
    validation_df[path] = validation_df.apply(lambda x: result.array[result.id_to_index[x["from"]],result.id_to_index[x["to"]]], axis=1)
    
    y_true = list(validation_df["close"].values*1)
    y_prob = list(1-validation_df[path].values)
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f_beta = lambda pr,re,beta: [((1+beta**2)*p*r)/((((beta**2)*p)+r)) for p,r in zip(pr,re)]
    f_1_scores = f_beta(precision,recall,beta=1)
    f_1_max = np.nanmax(f_1_scores)
    row_items.append(f_1_max)
    
    
    # Use the pairwise interface to actually get a set of document vectors and calculate maximum F_1 values.
    validation_df = po_validation_df
    result = pw.with_word2vec(model, po_key_to_preprocessed_text_string, "cosine", "mean")
    validation_df[path] = validation_df.apply(lambda x: result.array[result.id_to_index[x["from"]],result.id_to_index[x["to"]]], axis=1)
    
    y_true = list(validation_df["close"].values*1)
    y_prob = list(1-validation_df[path].values)
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f_beta = lambda pr,re,beta: [((1+beta**2)*p*r)/((((beta**2)*p)+r)) for p,r in zip(pr,re)]
    f_1_scores = f_beta(precision,recall,beta=1)
    f_1_max = np.nanmax(f_1_scores)
    row_items.append(f_1_max)
    
    
    # For the word similarity test.
    row_items.extend(["{}: {}".format(w, "; ".join([x[0] for x in model.most_similar(w,topn=10)])) for w in test_words])
    
    # Adding these results to a list to build a dataframe from.
    rows.append(tuple(row_items))
    
    
    
# Constructing and saving the results dataframe.
pd.DataFrame(rows, columns=header).to_csv(output_path_for_results, index=False)

