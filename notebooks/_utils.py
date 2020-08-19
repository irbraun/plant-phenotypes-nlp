import pandas as pd
import re




class Method:
	def __init__(self, name, hyperparameters, group, number, function=None, kwargs=None, metric=None, tag=""):
		"""Constructor for a Method class to define a single approach for distance measurements.
		
		Args:
		    name (str): A string specifying just the general name of the method, not including hyperparameter choices.
		    hyperparameters (str): A string specifying what the hyperparameter choices were for this method/model/approach.
		    group (str): Some additional description of what kind of category this approach belongds to.
		    number (int): Any integer, can be useful as a rank for sorting these methods when creating plots, because no other order is indicated anywhere.
		    function (Function): The function to use for obtaining the distance results, will be passed everything in kwargs.
		    kwargs (dict): A dictionary of all the arguments by keyword that should be passed to the function argument.
		    metric (scipy.spatial.distance.?): A distance metric form the scipy.spatial package.
		    tag (str, optional): An arbitrary string tag that can be used to associate some information with this method.
		
		Deleted Parameters:
		    data (str): A string that should be either 
		"""
		#self.name = name.lower()
		#self.hyperparameters = hyperparameters.lower()
		#self.name_with_hyperparameters = "{}:{}".format(name,hyperparameters).lower()

		self.name = re.sub('[^0-9a-zA-Z]+', '_', name.lower())
		self.hyperparameters = re.sub('[^0-9a-zA-Z]+', '_', hyperparameters.lower())
		self.name_with_hyperparameters = "{}__{}".format(self.name, self.hyperparameters)
		self.group = re.sub('[^0-9a-zA-Z]+', '_', group.lower())
		self.number = number
		self.function = function
		self.kwargs = kwargs
		self.metric = metric
		self.tag = tag











class IndexedGraph:

	def __init__(self, df):
		"""Constructor for an IndexedGraph class that indexes a dataframe of a specific shape.
		Args:
		    df (pandas.DataFrame): A dataframe with columns {from, to, method1, method2, ...}.
		"""
		self.ids_in_graph = self._get_ids_in_graph(df)
		self.indexed_df = self._index_df(df)

	def get_ids_in_graph(self):
		return(self.ids_in_graph)

	def get_value(self, id1, id2, kind):
		"""
		Get the value of an edge of a certain kind between two nodes. Note that this class
		is for an undirected graph, so the ordering of the IDs passed in to this function 
		does not matter. Both orderings are tried in finding a relevant edge weight to 
		return. The undirected nature of the graph is not actually enforced, so if two 
		directed edges are found for this node pair, the first one observed when searching
		is returned with no error raised.
		
		Args:
		    id1 (int): An ID for a node in the graph.
		    id2 (int): An ID for a node in the graph.
		    kind (str): The name a column for a certain kind of edge in the graph.
		Returns:
		    float: The weight of the edge between those two nodes.
		Raises:
		    KeyError: Could not find the edge in question.
		"""
		try:
			return(self.indexed_df.loc[(id1,id2)][kind])
		except(KeyError, TypeError):
			pass
		try: 
			return(self.indexed_df.loc[(id2,id1)][kind])
		except(KeyError, TypeError):
			raise KeyError("no value of this kind for this pair of IDs")

	def get_values(self, id_pairs, kind):
		"""
		Get the values of a list of edges between a list of pairs of nodes. Note that this 
		class is for an undirected graph, so the orderings of the IDs in each pair of node
		IDs passed in does not matter. However, this function works by simply getting all 
		rows in the internal dataframe which exactly match (i,j) in the (from,to) columns. 
		Therefore, all permutations of the desired edges should be passed in in order to 
		guarantee that the desired edges are found. The length of the returned list of edge
		weights may not necessarily match the length of the list of node ID pairs passed in.
		This is because this method silently removes from the list any pair that doesn't 
		have an entry in the internal dataframe. The intended use is to very quickly return
		a group of edge weights where the mapping between the passed in node pairs and the
		returned weights is not needed.
		
		Args:
		    id_pairs (list): List of tuples that contain exactly two integers (node IDs).
		    kind (str): The name of a column for a certain kind of edge in the graph.
		
		Returns:
		    list: A list of floats, which are the weights of all matching found edges.
		"""
		values = self.indexed_df[self.indexed_df.index.isin(id_pairs)][kind].values
		return(values)


	def _get_ids_in_graph(self, df):
		ids = pd.unique(df[["from","to"]].values.ravel('K'))
		return(ids)

	def _index_df(self, df):
		indexed_df = df.set_index(["from","to"], inplace=False)
		return(indexed_df)





