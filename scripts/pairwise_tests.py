

# this is very fast

# https://towardsdatascience.com/overview-of-text-similarity-metrics-3397c4601f50
def get_cosine_sim_matrix(*strs):
	vectors = [t for t in get_count_vectors(*strs)]
	similarity_matrix = cosine_similarity(vectors)
	return similarity_matrix




# Relatively this is very slow.
def get_jaccard_sim_matrix(*strs):
	vectors = [t for t in get_binary_vectors(*strs)]
	dist = DistanceMetric.get_metric("jaccard")

	a = time.perf_counter()
	similarity_matrix = dist.pairwise(vectors)
	b = time.perf_counter()
	print("Time for running dist.pairwise(vectors) was ", b-a)

	similarity_matrix = 1 - similarity_matrix
	return similarity_matrix












def get_cosine_sim_matrix(*strs):
	vectors = strings_to_count_vectors(*strs)
	print(np.histogram(vectors))
	print("The shape of the vectors for bow",np.array(vectors).shape)
	similarity_matrix = cosine_similarity(vectors)
	return(similarity_matrix)

def get_jaccard_sim_matrix(*strs):
	vectors = strings_to_binary_vectors(*strs)
	print(np.histogram(vectors))
	print("The shape of the vectors for sow",np.array(vectors).shape)
	matrix = pdist(vectors, "jaccard")
	matrix = squareform(matrix)
	similarity_matrix = matrix
	similarity_matrix = 1 - similarity_matrix
	return(similarity_matrix)





# DistanceMetric.pairwise(vectors) is very very slow (atleast when the DistanceMetric initilized as jaccard)
# Have to use something else.







# put this in a test or something


"""
import gensim
import itertools
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import cosine_similarity

n = 10000
object_dict = range(n)
vectors = []
for i in range(int(n)):
	vec = np.random.uniform(low=0.00, high=1.00, size=(300,))
	vectors.append(vec)


print("there are",len(object_dict),"ids")
print("there are",len(vectors),"vectors of length",len(vectors[1]))



# Using the scipy pdist library.
start = time.perf_counter()
matrix = pdist(vectors, "cosine")
matrix = squareform(matrix)
total = time.perf_counter()-start
print(total)

# The efficient way to do this...
df_of_matrix = pd.DataFrame(matrix)						# Convert numpy array to pandas dataframe.
df_of_matrix.reset_index(level=0, inplace=True) 		# Add the row indices as their own column.
melted_matrix = pd.melt(df_of_matrix,id_vars="index")	# Melt the adjacency matrix to get {from,to,value} form.
melted_matrix.columns = ["from", "to", "value"]			# Rename the columns to indicate this specifies a graph.

print(melted_matrix)


sys.exit()




# Build the dataframe representation of the pairwise similarities found for this method.
start = time.perf_counter()
result = pd.DataFrame(columns=["from", "to", "similarity"])
for (p1, p2) in list(itertools.combinations_with_replacement(object_dict, 2)):	
	row = [p1, p2, matrix[p1][p2]]
	result.loc[len(result)] = row
print(result)
total = time.perf_counter()-start
print(total)



sys.exit()


# Using the sklearn library
start = time.perf_counter()
matrix = cosine_similarity(vectors)
total = time.perf_counter()-start
print(total)

# Build the dataframe representation of the pairwise similarities found for this method.
result = pd.DataFrame(columns=["from", "to", "similarity"])
for (p1, p2) in list(itertools.combinations_with_replacement(object_dict, 2)):	
	row = [p1, p2, matrix[p1][p2]]
	result.loc[len(result)] = row
print(result)









#matrix = cosine_similarity(vectors)















sys.exit()
"""







'''
# https://towardsdatascience.com/overview-of-text-similarity-metrics-3397c4601f50
def get_cosine_sim_matrix(*strs):
	vectors = [t for t in get_count_vectors(*strs)]


	print("The shape of the vectors for bow",np.array(vectors).shape)



	similarity_matrix = cosine_similarity(vectors)
	return similarity_matrix

def get_count_vectors(*strs):
	text = [t for t in strs]
	vectorizer = CountVectorizer(text)
	vectorizer.fit(text)
	return vectorizer.transform(text).toarray()


def get_jaccard_sim_matrix(*strs):
	vectors = [t for t in get_binary_vectors(*strs)]
	dist = DistanceMetric.get_metric("jaccard")

	print("The shape of the vectors for sow",np.array(vectors).shape)

	#a = time.perf_counter()
	#similarity_matrix = dist.pairwise(vectors)
	#b = time.perf_counter()
	#print("Time for running dist.pairwise(vectors) was ", b-a)
	
	a = time.perf_counter()
	matrix = cosine_similarity(vectors)
	#matrix = pdist(vectors, "cosine")
	#matrix = squareform(matrix)
	b = time.perf_counter()
	print("With pdist, time for running dist.pairwise(vectors) was ", b-a)
	similarity_matrix = matrix

	similarity_matrix = 1 - similarity_matrix
	return similarity_matrix

def get_binary_vectors(*strs):
	text = [t for t in strs]
	#vectorizer = HashingVectorizer(text)
	vectorizer = TfidfVectorizer(text)
	vectorizer.fit(text)
	return vectorizer.transform(text).toarray()
'''

"""
https://datascience.stackexchange.com/questions/22250/what-is-the-difference-between-a-hashing-vectorizer-and-a-tfidf-vectorizer

The problem is that the hashing vectorizer is doing what it's supposed to.
Maps to huge feature space, default size is 2^20?
That could be reduced but then there's greater risk of collision (would have to calculate what that is).

Could probably try out the TfidfVectorizer as well, penalizing importance of features (words) if they are present
in lots of documents. Changing how the conversion to a numerical vector is done is another variable that can be 
changed along with changing what similarity metric is used once the vectors are formed.  TfidVectorizer also 
generates vectors that are the minimal size possible (size of the vocabulary), just like the CountVectorizer.

CountVectorizer is so good in this case because it used the minimal vector size (matches size of vocabulary)
in the dataset for every new problem. Therefore to get the binary count vector for Set of Words, just convert
the CountVectorized vectors so that 0-->0 and (1 or >1)-->1. 
"""


