from sklearn.metrics import precision_recall_curve
import itertools
import numpy as np








def classification(graph, id_to_labels, label_to_ids):

	# What labels (pathway IDs) and object IDs are specified in the mapping?
	labels = label_to_ids.keys()
	ids = id_to_labels.keys()

	# Construct the binary vector fo
	y_true = []
	y_scores = []
	for identifier, label in itertools.product(ids,labels):
		
		# What is the expected response, positive or negative?
		y_true.append(int(label in id_to_labels[identifier]))

		# What's the probability assigned to this classification based on the graph?
		ids_from_this_class = label_to_ids[label]
		ids_from_this_class = [x for x in ids_from_this_class if x is not identifier]

		# There are other things with this label to obtain a mean similarity with.
		if len(ids_from_this_class)>0:
			in_group_similarities = np.asarray([graph.get_value(identifier,i) for i in ids_from_this_class])
			in_group_similarities[np.isnan(in_group_similarities)] = 0.000
			in_group_mean = np.mean(in_group_similarities)
			y_scores.append(in_group_mean)

		# This is the only thing with that label anyway so can't obtain a mean.
		else:
			y_scores.append(0.000)

	return(y_true, y_scores)








def consistency_index(graph, id_to_labels, label_to_ids):
	
	# Want to return a dict of labels --> single value.
	labels = label_to_ids.keys()
	ids = id_to_labels.keys()

	label_to_consistency = {}


	for label in label_to_ids.keys():

		ids_from_this_class = label_to_ids[label]
		in_group_similarities = []
		out_group_similarities = []
		for identifier in ids_from_this_class:
			other_ids_from_this_class = [x for x in ids_from_this_class if x is not identifier]
			outside_ids = [x for x in ids if x not in ids_from_this_class]
			in_group_similarities.extend([graph.get_value(identifier,i) for i in other_ids_from_this_class])
			out_group_similarities.extend([graph.get_value(identifier,i) for i in outside_ids])

		ingroup = np.asarray(in_group_similarities)
		outgroup = np.asarray(out_group_similarities)

		if len(ingroup)>0 and len(outgroup)>0:
			ingroup[np.isnan(ingroup)] = 0.000
			outgroup[np.isnan(outgroup)] = 0.00
			ci = np.mean(ingroup) - np.mean(outgroup)
			label_to_consistency[label] = ci
		else:
			label_to_consistency[label] = 99.999

	return(label_to_consistency)





























def pr_curve(y_true, y_scores):
	precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
	print(precision)
	print(recall)
	print(thresholds)


	# graph has {from, to, similarity}
	# class_dict maps arbitrary class or group names to a list of IDs that have that label.


	# CHECKS
	# step 1. check to make sure all IDs in the label dict are present in the graph.
	# Step 2. check to make sure the graph is complete?




	'''
	>>> import numpy as np
	>>> from sklearn.metrics import precision_recall_curve
	>>> y_true = np.array([0, 0, 1, 1])
	>>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
	>>> precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
	>>> precision  
	array([0.66666667, 0.5       , 1.        , 1.        ])
	>>> recall
	array([1. , 0.5, 0.5, 0. ])
	>>> thresholds
	array([0.35, 0.4 , 0.8 ])
	'''