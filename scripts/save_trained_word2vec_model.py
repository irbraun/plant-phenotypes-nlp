import gensim



# Things that are the most likely to change.
input_text_path = "../data/corpus_related_files/untagged_text_corpora/phenotypes_all.txt"
output_model_path = "../models/plants_sg/word2vec.model"
vector_size = 300
epochs = 500


sentences = gensim.models.word2vec.Text8Corpus(input_text_path)


# Defining the size of the hidden layer and output layers as well as training parameters and training the model.
model = gensim.models.word2vec.Word2Vec(sentences,
	size=vector_size, 
	iter=epochs,
	window=10, 
	sg=1, 
	hs=1,
	workers=16, 
	alpha=0.025, 
	sample=1e-3)


# Saving the trained model to a file.
model.save(output_model_path)
print("done saving the model")




# Checking to make sure the model can be loaded and used for looking up embeddings.
model = gensim.models.Word2Vec.load(output_model_path)
a_word_in_vocab = list(model.wv.vocab.keys())[0]
vector = model[a_word_in_vocab]
print(len(vector))
print(vector)
print("done testing the model") 
