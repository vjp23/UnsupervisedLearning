# Vince Petaccio

# Import libraries
import sys
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as EM
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import SparseRandomProjection
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from yummy import rainbow

# Set the random seed
seed = 34
# Initialize the random seed as defined above
np.random.seed(seed)

# Prepare the datasets ---------------------------------------------------------------------------------------
def preprocess(dataset):
	# Determine which data to read in
	if dataset == 'forest_type':
		# Forest type classification data
		set_name = 'Forest'
		# Read in training data
		train = pd.read_csv('forest_type_train.csv')
		# Define the training data and labels
		x_train = train.iloc[:, :-1].values
		y_train = np.ravel(train['class'])
		# Read in test data 
		test = pd.read_csv('forest_type_test.csv')
		x_test = test.iloc[:, :-1].values
		y_test = np.ravel(test['class'])
		df_data = train
	elif dataset == 'cover_type':
		# Tree coverage type classification data
		set_name = 'Cover'
		# Read in the data
		cov_data = pd.read_csv('cover_type_train_500k.csv')
		cov_data_test = pd.read_csv('cover_type_test_81k.csv')
		# Seperate the training data from its labels
		x_train = cov_data.iloc[:, :-1].values
		y_train = cov_data.iloc[:, -1].values
		x_test = cov_data_test.iloc[:, :-1].values
		y_test = cov_data_test.iloc[:, -1].values
		df_data = cov_data
	else:
		# ERROR- invalid data set selection
		sys.exit('ERROR: dataset must be forest_type or cover_type.')
	# Convert labels to integers
	ltr_ind = 0
	lettermap = []
	for letter in set(y_test):
		y_test[y_test == letter] = ltr_ind
		y_train[y_train == letter] = ltr_ind
		ltr_ind += 1
		lettermap.append(letter)
	# Convert labels to one-hot encoding
	n_classes = max(max(y_train), max(y_test)) + 1
	y_train_one_hot = np_utils.to_categorical(y_train)
	y_test_one_hot = np_utils.to_categorical(y_test)
	# Ensure that the train and test set labels have the same number of classes
	while y_train_one_hot.shape[1] < y_test_one_hot.shape[1]:
		y_train_one_hot = np.hstack((y_train_one_hot, 
			np.zeros(y_train_one_hot.shape[0]).reshape(y_train_one_hot.shape[0], 1)))
	while y_test_one_hot.shape[1] < y_train_one_hot.shape[1]:
		y_test_one_hot = np.hstack((y_test_one_hot, 
			np.zeros(y_test_one_hot.shape[0]).reshape(y_test_one_hot.shape[0], 1)))
	# Return the datasets
	return [x_train, x_test, y_train, y_test, y_train_one_hot, 
	y_test_one_hot, df_data, n_classes, set_name, lettermap]

# Parallel Coordinates Plot ----------------------------------------------------------------------------------
def parallel_plot(df_data, n_classes, set_name, y=0):
	# Check the data type of the input data
	if type(df_data) != pd.core.frame.DataFrame:
		# Input data is not a dataframe- check for labels
		columns = list(np.arange(df_data.shape[1]))
		columns.append('class')
		# Check wheter a y vector was supplied
		if type(y) != int:
			# Ensure that y is a Numpy array
			if type(y) != np.ndarray:
				y = np.array(y)
			# First check whether y values are one-hot encoded
			if len(y[0].shape):
				# Convert one-hot to  integers
				y = np.array([np.argmax(s) for s in y])
			# Add the labels
			df_data = np.hstack((df_data, y.reshape(df_data.shape[0], 1)))
		# Define the dataframe
		df_data = pd.DataFrame(df_data, columns=columns)
	cols = rainbow(n_classes)
	data_norm = (df_data - df_data.min())/ (df_data.max() - df_data.min())
	data_norm['class'] = df_data['class']
	if data_norm.shape[0] > 500:
		data_norm = data_norm.sample(n=500)
	fig, ax = plt.subplots(figsize=(16, 7), dpi=100)
	pd.plotting.parallel_coordinates(data_norm, 'class', ax=ax, color=cols)
	leg = ax.legend(fontsize=16)
	leg.set_title('Class',prop={'size':16})
	ax.tick_params(labelsize=16, rotation=15)
	ax.set_xlabel('Feature', fontsize=20)
	ax.set_ylabel('Normalized Feature Value', fontsize=20)
	ax.set_title((set_name + ' Type Data Features'), fontsize=26)
	plt.show()

# k-Means Clustering -----------------------------------------------------------------------------------------
def kmeans(x, n_classes, min_iterations=25):
	kmeans = KMeans(n_clusters=n_classes, n_init=min_iterations).fit(x)
	return kmeans.labels_

# EM Clustering ----------------------------------------------------------------------------------------------
def em(x, n_classes, min_iterations=25):
	em = EM(n_components=n_classes, n_init=25).fit(x)
	return em.predict(x)

# PCA Feature Selection --------------------------------------------------------------------------------------
def pca(x, n_components=3):
	pca_obj = PCA(n_components=n_components)
	pca_obj.fit(x)
	return pca_obj.transform(x)

# ICA Feature Selection --------------------------------------------------------------------------------------
def ica(x, n_components=3):
	ica_obj = FastICA(n_components=n_components)
	ica_obj.fit(x)
	return ica_obj.transform(x)

# Sparse Randomized Project Feature Selection ----------------------------------------------------------------
def rp(x, n_components=3):
	rp_obj = SparseRandomProjection(n_components=n_components)
	return rp_obj.fit_transform(x)

# Linear Discriminant Analysis Feature Selection -------------------------------------------------------------
def lda(x, y, n_components=3):
	lda_obj = LDA(n_components=n_components)
	return lda_obj.fit_transform(x, y)

# Artificial neural network ----------------------------------------------------------------------------------
def ann(x_train, x_test, y_train_one_hot, y_test_one_hot, lettermap, n_classes, set_name, show_plots=True, 
	save_plots=False, save_name='ann_plot', plot_size=(16, 7), plot_text_sizes=[16, 20, 26], verbose_level=0):
	# Set architecture based upon dataset (chosen from gridsearch results)
	if set_name == 'Forest':
		ann_standardize = True
		hidden_layers = [16, 8]
		dropout = 0.5
		learning_rate = 0.001
		epochs = 100
		batch_size = 128
		activation = 'tanh'
	else:
		ann_standardize = True
		hidden_layers = [64, 32, 16]
		dropout = 0.5
		learning_rate = 0.0001
		epochs = 100
		batch_size = 512
		activation = 'sigmoid'
	# Standardize the data if requested
	if ann_standardize:
		# First, define the scaler
		scaler = StandardScaler().fit(x_train)
		# Scale the training data set
		x_train = scaler.transform(x_train)
		# Scale the test set
		x_test = scaler.transform(x_test)

	# Build the neural network model --------------------------------------------------------------
	model = Sequential()
	# Add an input layer
	model.add(Dense(hidden_layers[0], input_shape=(len(x_test[0]),)))
	model.add(BatchNormalization())
	model.add(Activation(activation))
	model.add(Dropout(dropout))
	# Iteratively add model layers with batch normalizatio and dropout
	for layer_nodes in hidden_layers[1:]:
		model.add(Dense(layer_nodes))
		model.add(BatchNormalization())
		model.add(Activation(activation))
		model.add(Dropout(dropout))
	# Add an output layer with softmax activation for probabilities
	model.add(Dense(n_classes, activation='softmax'))
	# Define loss function as categorical cross entropy and ADAM optimizer
	optimizer = optimizers.Adam(lr=learning_rate)
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

	# Train the model -----------------------------------------------------------------------------
	checkpointer = ModelCheckpoint(filepath='weights.best.hdf5', verbose=0, save_best_only=True)
	history_train = model.fit(x_train, y_train_one_hot, batch_size=batch_size, epochs=epochs, 
		verbose=verbose_level, validation_data=(x_test, y_test_one_hot), callbacks=[checkpointer])
	# Load the best weights encountered during training for later evaluation
	model.load_weights('weights.best.hdf5')

	# Get the accuracy data from the model after it's done running --------------------------------
	history_test = history_train.history['acc']
	history_val = history_train.history['val_acc']
	history_loss = history_train.history['loss']
	# Plot the training and validation accuracy curves
	f, a = plt.subplots(figsize=plot_size, dpi=100)
	x = range(len(history_test))
	a.plot(x, history_test, color='#7d1df8', label='Training Accuracy')
	a.plot(x, history_val, color='#da008a', label='Validation Accuracy')
	# Format the plot
	a.legend(fontsize=plot_text_sizes[0])
	a.set_xlim([0, epochs])
	a.tick_params(labelsize=plot_text_sizes[0])
	a.set_xlabel('Training Epoch', fontsize=plot_text_sizes[1])
	a.set_ylabel('Accuracy', fontsize=plot_text_sizes[1])
	a.set_title('Training and Validation Accuracy- Neural Network on {} Data'.format(set_name), 
		fontsize=plot_text_sizes[2])
	# Save the figure
	if save_plots:
		plt.savefig(save_name)
		print('Figure saved to', save_name)

	# Perform a prediction and map probabilites to labels -----------------------------------------
	y_pred = model.predict(x_test)
	y_pred_labels = [lettermap[np.argmax(s)] for s in y_pred]
	score = model.evaluate(x_test, y_test_one_hot, verbose=0)
	# Print loss and accuracy
	print('ANN Results for {} data:'.format(set_name.lower()))
	print('Validation accuracy peaked at {} after {} epochs, with a loss of {}.'.format(max(history_val), 
		np.argmax(np.array(history_val)) + 1, history_loss[np.argmax(np.array(history_val))]))

	if show_plots:
		# Display the plot figure
		plt.show()

# Run the experiments ----------------------------------------------------------------------------------------
def run_experiments(experiment=0, data='forest_type'):
	# Validate the experiment input
	if experiment not in [0, 1, 2, 3, 4, 5]:
		sys.exit('Choose an experiment number from 1 to 5, or use 0 to run all experiments.')
	# Validate the data input
	if data != 'forest_type' and data != 'cover_type':
		sys.exit('Choose the data set to use, either "forest_type" or "cover_type".')

	# Load and preprocess the data
	print('\nLoading', data, 'data...')
	[x_train, x_test, y_train, y_test, y_train_one_hot, y_test_one_hot, 
	 df_data, n_classes, set_name, lettermap] = preprocess(data)

	if experiment == 0 or experiment == 1:
		# Perform the clustering
		print('\nPerforming clustering...\n')
		k_labels_train = kmeans(x_train, n_classes)
		print('K-Means clustering done.')
		em_labels_train = em(x_train, n_classes)
		print('Expectation maximization clustering done.\n')

		# Determine the accuracy for each method
		accuracy = []
		for cluster_set in [k_labels_train, em_labels_train]:
			modes = [stats.mode(cluster_set[np.where(y_train==list(set(y_train))[j])])[0][0] for j in range(len(list(set(y_train))))]
			indices = []
			for label in modes:
				mode_idx = list(list(np.where(cluster_set == label))[0])
				indices.append(mode_idx)
			for value, index in zip(list(set(y_train)), indices):
				cluster_set[index] = value
			accuracy.append(np.sum(y_train==cluster_set) / y_train.shape[0])

		# Return the results
		print('Accuracy for K-means clustering on the', set_name, 'data was', str(accuracy[0]) + '.')
		print('Accuracy for EM clustering on the', set_name, 'data was', str(accuracy[1]) + '.')

	if experiment == 0 or experiment == 2:
		# Perform dimensionality reduction and save 3D plots. First, subsample the cover data set
		if x_train.shape[1] > 500:
			rand_rows = np.random.choice(x_train.shape[0], size=500, replace=False)
			x_small = x_train[rand_rows, :]
			y_small = y_train[rand_rows]
		else:
			x_small = x_train
			y_small = y_train
		
		# Normalize the data before displaying
		x_normed = (x_small - x_small.min(0)) / x_small.ptp(0)
	
		# Do the feature reduction to 3 features
		x_pca = pca(x_normed)
		x_ica = ica(x_normed)
		x_rp = rp(x_normed)
		x_lda = lda(x_normed, y_small)

		# Create the 3D plots and save them
		for x_t, algo in zip([x_pca, x_ica, x_rp, x_lda], ['_PCA_3D', '_ICA_3D', '_RP_3D', '_LDA_3D']):
			fig = plt.figure(figsize=(16, 7), dpi=100)
			ax = fig.add_subplot(111, projection='3d')
			cols = rainbow(colors=n_classes, frac=0.7)
			for label in range(n_classes):
				examples = np.where(y_small == label)
				data = x_t[examples, :]
				x = data[:, :, 0]
				y = data[:, :, 1]
				z = data[:, :, 2]
				ax.scatter(x, y, z, color=cols[label])
			ax.set_title('Cover Type Data, PCA with 3 Components', fontsize=26)
			ax.set_xlabel('Component 1', fontsize=20, labelpad=20)
			ax.set_ylabel('Component 2', fontsize=20, labelpad=20)
			ax.set_zlabel('Component 3', fontsize=20, labelpad=20)
			ax.tick_params(labelsize=16)
			plt.savefig(set_name + algo)
			print('Saved ' + set_name + algo + '.png')

	if experiment == 0 or experiment == 3:
		# Normalize the data before training
		x_normed = (x_train - x_train.min(0)) / x_train.ptp(0)

		# Do the feature reduction to 4 features
		x_pca = pca(x_normed, 4)
		x_ica = ica(x_normed, 4)
		x_rp = rp(x_normed, 4)
		x_lda = lda(x_normed, y_train, 4)

		# Perform the clustering for each result
		for x_reduced, algo in zip([x_pca, x_ica, x_rp, x_lda], ['PCA', 'ICA', 'RP', 'LDA']):
			print('\nPerforming clustering on', algo, 'data...\n')
			k_labels_train = kmeans(x_reduced, n_classes)
			print('K-Means clustering done.')
			em_labels_train = em(x_reduced, n_classes)
			print('Expectation maximization clustering done.\n')

			# Determine the accuracy for each method
			accuracy = []
			for cluster_set in [k_labels_train, em_labels_train]:
				modes = [stats.mode(cluster_set[np.where(y_train==list(set(y_train))[j])])[0][0] for j in range(len(list(set(y_train))))]
				indices = []
				for label in modes:
					mode_idx = list(list(np.where(cluster_set == label))[0])
					indices.append(mode_idx)
				for value, index in zip(list(set(y_train)), indices):
					cluster_set[index] = value
				accuracy.append(np.sum(y_train==cluster_set) / y_train.shape[0])

			# Return the results
			print('Accuracy for K-means clustering on the', set_name, 'data with', algo, 'was', str(accuracy[0]) + '.')
			print('Accuracy for EM clustering on the', set_name, 'data with', algo, 'was', str(accuracy[1]) + '.')

	if experiment == 0 or experiment == 4:
		# Normalize the data
		x_train = (x_train - x_train.min(0)) / x_train.ptp(0)
		x_test = (x_test - x_test.min(0)) / x_test.ptp(0)

		# Perform the feature reduction
		print('\nData loaded. Begin feature reduction to 4 features...\n')
		x_pca = pca(x_train, 4)
		x_test_pca = pca(x_test, 4)
		print('Principal component analysis done.')
		x_ica = pca(x_train, 4)
		x_test_ica = pca(x_test, 4)
		print('Independent component analysis done.')
		x_rp = rp(x_train, 4)
		x_test_rp = rp(x_test, 4)
		print('Sparse randomized projection done.')
		x_lda = lda(x_train, y_train, 4)
		x_test_lda = lda(x_test, y_test, 4)
		print('Linear discriminant analysis done.')

		# Run the ANN on each set of features
		print('\nFeature reduction complete. Begin running ANNs...\n')
		print('\nRunning ANN on principal component analysis features...\n')
		ann(x_pca, x_test_pca, y_train_one_hot, y_test_one_hot, lettermap, n_classes, 
			set_name, show_plots=False, save_plots=True, save_name=data+'_pca_ann')
		print('\nRunning ANN on independent component analysis features...')
		ann(x_ica, x_test_ica, y_train_one_hot, y_test_one_hot, lettermap, n_classes, 
			set_name, show_plots=False, save_plots=True, save_name=data+'_ica_ann')
		print('\nRunning ANN on randomized projection features...')
		ann(x_rp, x_test_rp, y_train_one_hot, y_test_one_hot, lettermap, n_classes, 
			set_name, show_plots=False, save_plots=True, save_name=data+'_rp_ann')
		print('\nRunning ANN on linear discriminant analysis features...')
		ann(x_lda, x_test_lda, y_train_one_hot, y_test_one_hot, lettermap, n_classes, 
			set_name, show_plots=False, save_plots=True, save_name=data+'_lda_ann')
		print('\nDone running ANNs.\n')

	if experiment == 0 or experiment == 5:
		# Perform the clustering, and add the clusters to the data as new features.
		print('\nPerforming clustering...\n')
		k_labels_train = kmeans(x_train, n_classes)
		k_labels_test = kmeans(x_test, n_classes)
		print('K-Means clustering done.')
		em_labels_train = em(x_train, n_classes)
		em_labels_test = em(x_test, n_classes)
		print('Expectation maximization clustering done.')

		# Add clusters to dataset as features
		train_clusters = np.vstack((k_labels_train, em_labels_train))
		test_clusters = np.vstack((k_labels_test, em_labels_test))
		x_train_cluster = np.hstack((x_train, train_clusters.reshape(x_train.shape[0], 2)))
		x_test_cluster = np.hstack((x_test, test_clusters.reshape(x_test.shape[0], 2)))

		# Run the ANN on each set of features
		print('\nBegin running ANN on clustered data...\n')
		ann(x_train_cluster, x_test_cluster, y_train_one_hot, y_test_one_hot, lettermap, n_classes, 
			set_name, show_plots=False, save_plots=True, save_name=data+'_clustered')
		print('\nDone running ANN on clustered data.\n')