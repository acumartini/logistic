# Adam Martini
# Utilities for data preprocessing/management and classification evaluation.
# mlutils.py

import numpy as np
import h5py

def load_csv(data, shuffle=False):
	"""
	Loads the csv files into numpy arrays.
	@parameters: data The data file in csv format to be loaded
				 shuffle True => shuffle data instances to randomize
	@returns: X - numpy array of data instances with dtype=float
			  y - numpy array of labels
	"""
	print("Loading data from", data)
	dset = np.loadtxt(data, delimiter=",", dtype='float')
	return shuffle_split(dset, shuffle)

def saveh(dset, path):
	"""
	Stores the numpy data in h5 format.
	@parameters: dset Dataset to store
				 path The path to file (including the file name) to save h5 file to
	"""
	f = h5py.File(path, 'w')
	f['dset'] = dset
	f.close()

def loadh(path, shuffle=False):
	"""
	Loads the h5 data into a numpy array.
	@parameters: path The path to file (including the file name) to load data from
				 shuffle True => shuffle data instances to randomize
	@returns: X - numpy array of data instances with dtype=float
			  y - numpy array of labels
	"""
	f = h5py.File(path,'r') 
	data = f.get('dset') 
	dset = np.array(data)
	return shuffle_split(dset, shuffle)

def shuffle_split(dset, shuffle):
	# randomize data
	if shuffle:
		dset = shuffle_data(dset)

	# split instances and labels
	y = dset[:,-1:] # get only the labels
	X = dset[:,:-1] # remove the labels column from the data array

	return X, y

def shuffle_data(X):
	# get a random list of indices
	rand_indices = np.arange(X.shape[0])
	np.random.shuffle(rand_indices)

	# build shuffled array
	X_ = np.zeros(X.shape)
	for i, index in enumerate(rand_indices):
		X_[i] = X[index]
	return X_

def mean_normalize(X, std=False):
	# normalize the mean to 0 for each feature and scale based on max/min values or
	# the standard deviation according to paramter "std"
	d = X.std(0) if std else X.max(0) - X.min(0)
	return (X - X.mean(0)) / d

def scale_features(X, new_min, new_max):
	# scales all features in dataset X to values between new_min and new_max
	X_min, X_max = X.min(0), X.max(0)
	return (((X - X_min) / (X_max - X_min)) * (new_max - new_min + 0.000001)) + new_min

def multiclass_format(y, c):
	"""
	Formats dataset labels y to a vector representation for multiclass classification.
	i.e., If there are 3 classes {0,1,2}, then all instances of 0 are transformed to
	[1,0,0], 1''s are tranformed to [0,1,0], and 2's become [0,0,1]
	"""
	y_ = np.zeros(shape=(len(y), c));
	for i, lable in enumerate(y):
		y_[i][int(lable)] = 1.0
	return y_

def compute_accuracy(y_test, y_pred):
	"""
	@returns: The precision of the classifier, (correct labels / instance count)
	"""
	correct = 0
	for i, pred in enumerate(y_pred):
		if int(pred) == y_test[i]:
			correct += 1
	return float(correct) / y_test.size