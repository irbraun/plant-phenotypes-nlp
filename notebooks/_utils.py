class Method:
	def __init__(self, name_str, hyperparameter_str, function, kwargs, metric, tag=""):
		"""Constructor for a Method class to define a single approach for distance measurements.
		
		Args:
		    name_str (str): A string specifying just the general name of the method, not including hyperparameter choices.
		    hyperparameter_str (str): A string specifying what the hyperparameter choices were for this method/model/approach.
		    function (Function): The function to use for obtaining the distance results, will be passed everything in kwargs.
		    kwargs (dict): A dictionary of all the arguments by keyword that should be passed to the function argument.
		    metric (scipy.spatial.distance.?): A distance metric form the scipy.spatial package.
		    tag (str, optional): An arbitrary string tag that can be used to associate some information with this method.
		
		Deleted Parameters:
		    data (str): A string that should be either 
		"""
		self.name = name_str
		self.hyperparameters = hyperparameter_str
		self.name_with_hyperparameters = "{}:{}".format(name_str,hyperparameter_str)
		self.function = function
		self.kwargs = kwargs
		self.metric = metric
		self.tag = tag