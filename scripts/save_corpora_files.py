from nltk.corpus import brown
import nltk
import sys
import time
nltk.download('punkt')
nltk.download('brown')

sys.path.append("../../oats")
from oats.pubmed.query import search, fetch_details





# Where to save the corpora files from either background or interesting text.
background_path = "../data/corpus_related_files/background.txt"
ofinterest_path = "../data/corpus_related_files/phenotypes_maize.txt"

# Getting a sampling of text that could be considered background or general information.
background_text = " ".join(brown.words(categories=['news',"editorial","reviews","lore"]))
background_file = open(background_path,"w")
background_file.write(background_text)
background_file.close()

# Getting a sampling of text that is from some domain of interest.
interesting_texts = []
limit = 1000000
fetch_batch_size = 1000
seconds_between_fetch = 5

query = "maize AND phenotype"
print("querying")
results = search(query, limit)
id_list = results['IdList']
print("{} abstracts found with this query".format(len(id_list)))
if len(id_list) > 0:
	print("fetching details")
	id_list_batches = [id_list[x:x+fetch_batch_size] for x in range(0, len(id_list), fetch_batch_size)]
	print("splitting into {} batches".format(len(id_list_batches)))
	for i,id_list_batch in enumerate(id_list_batches):
		print("batch {}".format(i))
		papers = fetch_details(id_list_batch)
		for i, paper in enumerate(papers['PubmedArticle']):
			try:
				abstract_text = paper['MedlineCitation']['Article']['Abstract']["AbstractText"][0]
				interesting_texts.append(abstract_text)
			except KeyError:
				continue
		time.sleep(seconds_between_fetch)
interesting_text = " ".join(interesting_texts)
ofinterest_file = open(ofinterest_path,"w")
ofinterest_file.write(interesting_text)
ofinterest_file.close()