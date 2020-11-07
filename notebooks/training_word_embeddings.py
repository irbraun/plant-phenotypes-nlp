#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import gensim
import random
import sys
from nltk import sent_tokenize
from nltk import word_tokenize
from scipy.spatial.distance import cosine
import warnings
from gensim.parsing.preprocessing import strip_non_alphanum, stem_text, preprocess_string
from gensim.utils import simple_preprocess
from gensim.models.callbacks import CallbackAny2Vec

warnings.simplefilter('ignore')

sys.path.append("../../oats")
from oats.annotation.ontology import Ontology
from oats.distances import pairwise as pw
from oats.utils.utils import flatten


# In[7]:


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





# In[57]:


# Checking to make sure the model can be loaded and used for looking up embeddings.
path = "../models/wiki_sg/word2vec.bin"
model = gensim.models.Word2Vec.load(path)
a_word_in_vocab = list(model.wv.vocab.keys())[0]
vector = model[a_word_in_vocab]
print(len(vector))


# In[ ]:





# In[11]:


# Checking to make sure the model can be loaded and used for looking up embeddings.
path = "../models/plants_sg/word2vec_ep500_dim150.model"
model = gensim.models.Word2Vec.load(path)
a_word_in_vocab = list(model.wv.vocab.keys())[0]
vector = model[a_word_in_vocab]
print(len(vector))
model[["small"]]


# In[62]:


list(model.wv.vocab)[10]

set_of_interest = ["auxin","root","shoot","growth","ear","kernel","maiz","length","wide","dwarf","maize","arabidopsis","species"]
a = list(np.random.choice(list(model.wv.vocab), replace=False, size=1000))
a = []
a.extend(set_of_interest)




arr = np.empty((0,300), dtype='f')
word_labels = []

colors = []
for w in a:
    if w in set_of_interest:
        colors.append(1)
    else:
        colors.append(0)

for w in a:
    wrd_vector = model[w]
    word_labels.append(w)
    arr = np.append(arr, np.array([wrd_vector]), axis=0)
        
        
# find tsne coords for 2 dimensions
tsne = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
Y = tsne.fit_transform(arr)

x_coords = Y[:, 0]
y_coords = Y[:, 1]
# display scatter plot
plt.scatter(x_coords, y_coords, c=colors)

for label, x, y in zip(word_labels, x_coords, y_coords):
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
plt.show()


# In[28]:


import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["figure.dpi"] = 400
 
from sklearn.manifold import TSNE


def display_closestwords_tsnescatterplot(model, word):
    
    arr = np.empty((0,150), dtype='f')
    word_labels = [word]

    # get close words
    close_words = model.similar_by_word(word, 30)
    
    # add the vector for each of the closest words to the array
    
    colors = []
    colors.append(1)
    for w in close_words:
        if w[0] == "zone":
            colors.append(1)
        else:
            colors.append(0)
    
    
    arr = np.append(arr, np.array([model[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)
        
    
        
    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display scatter plot
    plt.scatter(x_coords, y_coords, c=colors)

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    plt.show()
    
display_closestwords_tsnescatterplot(model, "root")


# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[2]:


# Load the ontology and term information.
path = "../ontologies/po.obo"
ont = Ontology(path)
term_ids_and_names = [(t.id,t.name) for t in ont.terms() if "obsolete" not in t.name]
key_to_annotations = {i:[x[0]] for i,x in enumerate(term_ids_and_names)}
key_to_term_id = {i:x[0] for i,x in enumerate(term_ids_and_names)}
key_to_text_string = {i:x[1] for i,x in enumerate(term_ids_and_names)}
key_to_preprocessed_text_string = {i:" ".join(preprocess_string(s)) for i,s in key_to_text_string.items()}


# In[3]:


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


# In[4]:


df = pw.with_annotations(key_to_annotations, ont, "jaccard", tfidf=False).edgelist
df = df[df["from"]!=df["to"]]
df["from_id"] = df["from"].map(lambda x: key_to_term_id[x])
df["to_id"] = df["to"].map(lambda x: key_to_term_id[x])
df["from_text"] = df["from"].map(lambda x: key_to_text_string[x])
df["to_text"] = df["to"].map(lambda x: key_to_text_string[x])
df["close"] = df.apply(lambda x: x["to_id"] in any_close[x["from_id"]], axis=1)
df["token_overlap"] = df.apply(lambda x: len(set(x["from_text"].split()).intersection(set(x["to_text"].split())))>0, axis=1)
df.head(20)


# In[5]:


df.shape


# In[6]:


positive_df = df[(df["token_overlap"]==False) & (df["close"]==True)]
negative_df = df[(df["token_overlap"]==False) & (df["close"]==False)]
assert negative_df.shape[0]+positive_df.shape[0] == df[df["token_overlap"]==False].shape[0]
num_positive_examples = positive_df.shape[0]
training_df = pd.concat([positive_df, negative_df.sample(num_positive_examples, random_state=2)])
del df
training_df.shape


# In[7]:


training_df.head(10)


# In[ ]:





# In[10]:


path = "../models/plants_sg/word2vec_ep500_dim150.model"
model = gensim.models.Word2Vec.load(path)
result = pw.with_word2vec(model, key_to_preprocessed_text_string, "cosine", "mean")
training_df["m"] = training_df.apply(lambda x: result.array[result.id_to_index[x["from"]],result.id_to_index[x["to"]]], axis=1)
training_df.head(10)


# In[11]:


from sklearn.metrics import precision_recall_curve
y_true = list(training_df["close"].values*1)
y_prob = list(1-training_df["m"].values)
precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
f_beta = lambda pr,re,beta: [((1+beta**2)*p*r)/((((beta**2)*p)+r)) for p,r in zip(pr,re)]
f_1_scores = f_beta(precision,recall,beta=1)
f_1_max = np.nanmax(f_1_scores)
f_1_max


# In[12]:


path = "../models/wiki_sg/word2vec.bin"
model = gensim.models.Word2Vec.load(path)
result = pw.with_word2vec(model, key_to_text_string, "cosine", "mean")
training_df["m"] = training_df.apply(lambda x: result.array[result.id_to_index[x["from"]],result.id_to_index[x["to"]]], axis=1)
training_df.head(10)

from sklearn.metrics import precision_recall_curve
y_true = list(training_df["close"].values*1)
y_prob = list(1-training_df["m"].values)
precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
f_beta = lambda pr,re,beta: [((1+beta**2)*p*r)/((((beta**2)*p)+r)) for p,r in zip(pr,re)]
f_1_scores = f_beta(precision,recall,beta=1)
f_1_max = np.nanmax(f_1_scores)
f_1_max


# In[13]:


training_df.sample(30)


# In[14]:


a = model[["driving"]]
b = model[["driver"]]
cosine(a,b)


# In[15]:


y_true = list(training_df["close"].values*1)
y_prob = list(1-training_df["m"].values)
random.shuffle(y_prob)
precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
f_beta = lambda pr,re,beta: [((1+beta**2)*p*r)/((((beta**2)*p)+r)) for p,r in zip(pr,re)]
f_1_scores = f_beta(precision,recall,beta=1)
f_1_max = np.nanmax(f_1_scores)
f_1_max


# In[16]:


print(setophere)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


max_retain_in_each_bin = len(df[df["bin"]==0.5])
df = df.groupby("bin", group_keys=False).apply(lambda x: x.sample(min(max_retain_in_each_bin,len(x))))
df.shape


# In[ ]:


path = "../models/plants_sg/word2vec_ep500_dim150.model"
model = gensim.models.Word2Vec.load(path)
model

result = pw.with_word2vec(model, key_to_text_string, "cosine", "mean")
df["thing"] = df.apply(lambda x: result.array[result.id_to_index[x["from"]],result.id_to_index[x["to"]]], axis=1)
df


# In[ ]:


from scipy.stats import pearsonr
pearsonr(df["value"],df["thing"])


# In[ ]:


key_to_preprocssed_text_string = {i:" ".join(preprocess_string(s)) for i,s in key_to_text_string.items()}
result = pw.with_word2vec(model, key_to_preprocssed_text_string, "cosine", "mean")
df["thing2"] = df.apply(lambda x: result.array[result.id_to_index[x["from"]],result.id_to_index[x["to"]]], axis=1)
df






# In[ ]:


from scipy.stats import pearsonr
from scipy.stats import spearmanr
spearmanr(df["value"],df["thing2"])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


df["from_id"] = df["from"].map(lambda x: annotations[x][0])
df["to_id"] = df["to"].map(lambda x: annotations[x][0])
df["from_text"] = df["from_id"].map(lambda x: term_ids_to_strings[x])
df["to_text"] = df["to_id"].map(lambda x: term_ids_to_strings[x])
df


# In[ ]:





# In[ ]:


path = "../models/plants_sg/word2vec_ep500_dim150.model"
model = gensim.models.Word2Vec.load(path)
model


# In[ ]:
















def f(from_text, to_text, model):
    cosine(pw.vectorize_with_word2vec(from_text, model, "max"),pw.vectorize_with_word2vec(from_text, model, "max"))




    
df["thing"] = df.apply(lambda row: f(row["from_text"],row["to_text"],model), axis=1)
df




# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


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
sentences_from_descriptions = list(np.repeat(sentences_from_descriptions, times_to_duplicate_phenotype_dataset))
sentences = sentences_from_corpus+sentences_from_descriptions
random.shuffle(sentences)
sentences = [preprocess_string(sentence) for sentence in sentences]
print(len(sentences))


# In[ ]:


print("starting training")


# In[ ]:





# In[ ]:


# Training the word2vec neural network with the current set of hyperparameters. 
model = gensim.models.Word2Vec(min_count=4, window=10, size=50, workers=4, alpha=0.025, min_alpha=0.025)
model.build_vocab(sentences)
loss_logger = LossLogger()
model.train(sentences, epochs=500, total_examples=model.corpus_count, compute_loss=True, callbacks=[loss_logger])

# Saving the model to a file.
output_path = "../models/plants_sg/word2vec_ep500_dim50.model"
model.save(output_path)
print("done training 1 ")


# In[ ]:


# Training the word2vec neural network with the current set of hyperparameters. 
model = gensim.models.Word2Vec(min_count=4, window=10, size=100, workers=4, alpha=0.025, min_alpha=0.025)
model.build_vocab(sentences)
loss_logger = LossLogger()
model.train(sentences, epochs=500, total_examples=model.corpus_count, compute_loss=True, callbacks=[loss_logger])

# Saving the model to a file.
output_path = "../models/plants_sg/word2vec_ep500_dim100.model"
model.save(output_path)
print("done training 2")


# In[ ]:


# Training the word2vec neural network with the current set of hyperparameters. 
model = gensim.models.Word2Vec(min_count=4, window=10, size=150, workers=4, alpha=0.025, min_alpha=0.025)
model.build_vocab(sentences)
loss_logger = LossLogger()
model.train(sentences, epochs=500, total_examples=model.corpus_count, compute_loss=True, callbacks=[loss_logger])

# Saving the model to a file.
output_path = "../models/plants_sg/word2vec_ep500_dim150.model"
model.save(output_path)
print("done training 3")


# In[ ]:


print(stophere)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


preprocess_string("plants leaves genes proteins tall wide abscisic root nodule")


# In[ ]:


from scipy.spatial.distance import cosine
print(cosine(model[["plant"]],model[["leav"]]))
print(cosine(model[["gene"]],model[["leav"]]))
print(cosine(model[["gene"]],model[["protein"]]))
print(cosine(model[["auxin"]],model[["hormon"]]))
print(cosine(model[["hormon"]],model[["abscis"]]))
print(cosine(model[["root"]],model[["nodul"]]))
len(model.wv.vocab)


# In[ ]:


# Checking to make sure the model can be loaded and used for looking up embeddings.
path = "../models/wiki_sg/word2vec.bin"
model_from_wikipedia = gensim.models.Word2Vec.load(path)
a_word_in_vocab = list(model_from_wikipedia.wv.vocab.keys())[0]
vector = model_from_wikipedia[a_word_in_vocab]
print(len(vector))


# In[ ]:


len(model.wv.vocab)


# In[ ]:


model.intersect_word2vec_format("../models/wiki_sg/word2vec.bin", binary=True, lockf=1.0)
model.train(sentences, epochs=1, total_examples=model.corpus_count, compute_loss=True, callbacks=[loss_logger])
len(model.wv.vocab)


# In[ ]:





# In[ ]:




