import gensim
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess
from nltk.tokenize import sent_tokenize



# Things that are the most likely to change.
input_text_path = "../data/corpus_related_files/untagged_text_corpora/phenotypes_all.txt"
output_model_path = "../models/plants_dbow/doc2vec.model"
vector_size = 300
epochs = 100





# Reading in a file of domain specific sentences and splitting into sentences.
pubmed_file = open(input_text_path, "r")
pubmed_text = pubmed_file.read()
sentences = sent_tokenize(pubmed_text)
print(len(sentences))
print(sentences[0])


# Defining the size of the hidden layer and output layers as well as training parameters.
model = gensim.models.Doc2Vec(
	vector_size=vector_size, 
	window=10, 
	min_count=5,
	dm=0, 
	workers=16, 
	alpha=0.025, 
	min_alpha=0.025, 
	dbow_words=1)

# Preprocessing the documents to be trained on and building the vocabulary.
tagged_docs = [TaggedDocument(words=simple_preprocess(s),tags=[str(i)]) for i,s in enumerate(sentences)]
model.build_vocab(tagged_docs)


# Training the neural network.
for epoch in range(epochs):
	model.train(tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)
	model.alpha -= 0.002            
	model.min_alpha = model.alpha
	if (epoch+1)%10==0:
		print("finished {} epochs".format(epoch+1))
model.save(output_model_path)
print("done saving the model")


# Checking to make sure the model can be loaded and used for the inference step.
model = gensim.models.Doc2Vec.load(output_model_path)
text = "some example text of something we want to embed"
vector = model.infer_vector(simple_preprocess(text))
print(len(vector))
print(vector)
print("done testing the model")
