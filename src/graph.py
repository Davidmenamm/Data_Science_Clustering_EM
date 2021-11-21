""" For all graphing functions """

# imports
from operator import index
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from sklearn import metrics, manifold as mf
from sklearn.decomposition import PCA
import seaborn as sns


# graph auc
def graphCurve(y_true, y_pred_proba, functionName, fileName, basePath, y_pred_categ, loss_epoch):
	# base folder
	baseFolder = basePath+f'\_{fileName}'
	# check if already exists
	if not os.path.exists(baseFolder):
		os.makedirs(baseFolder)
	# algorithm graphs
	filePath = f'{baseFolder}\\_TopResult_{functionName}.png'
	# give values
	fig = plt.figure(f'{fileName}-TopResult-{functionName}')
	plt.title(f'{fileName} - {functionName}')
	# manage cases
	graph = None
	if functionName == 'ROC':
		graph = metrics.RocCurveDisplay.from_predictions(y_true, y_pred_proba, name='TopResult', ax=plt.gca())
	elif functionName == 'Prec_Recall':
		graph = metrics.PrecisionRecallDisplay.from_predictions(y_true, y_pred_proba, name='TopResult', ax=plt.gca())
	elif functionName == 'Conf_Matrix':
		graph = metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_pred_categ, ax=plt.gca())
	elif functionName == 'Loss_Epoch':
		# epoch
		x_axis = range(1, len(loss_epoch)+1)
		# loss
		y_axis = loss_epoch
		# figure, axes
		plt.plot(x_axis, y_axis, 'b-', label = 'loss vs epochs')
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		# plt.legend(loc='best')
	# plot
	fig.savefig(filePath, dpi=fig.dpi)
	plt.clf()






# draw table
def drawTable(data, fileName, basePath):
	# base folder
	baseFolder = basePath+f'\_{fileName}'
	# check if already exists
	if not os.path.exists(baseFolder):
		os.makedirs(baseFolder)
	# algorithm graphs
	filePath = f'{baseFolder}\\_table.png'
	# get column names
	columns = ['Layers', 'Active', 'lr', 'epoch',
				'acc_avg', 'acc_std', 'prec_avg',
				'prec_std','rec_avg', 'rec_std',
				'auc_avg', 'auc_std','loss_avg',
				'loss_std']
	# counter	
	count = 0
	# get values
	cell_text = []
	for row in data:
		rowStr = []
		for value in row:
			# append
			rowStr.append(value)
		# append
		cell_text.append(rowStr)
		# increment
		count += 1
	# figure
	fig, ax = plt.subplots()
	fig.set_size_inches(20, 61)
	# hide axes
	fig.patch.set_visible(False)
	ax.axis('off')
	ax.axis('tight')
	# axis content
	labels_vertical = [str(lbl) for lbl in range(1, count+1)]
	ytable = ax.table(cellText = cell_text, rowLabels = labels_vertical, colLabels=columns, loc='center')
	ytable.set_fontsize(24)
	ytable.scale(1, 4)
	# plot
	fig.savefig(filePath, dpi=fig.dpi)
	fig.clf()




# determine which is the optimum k value
def bestK(name, outputFile, clusters, aic, bic):
	# base folder
	baseFolder = outputFile + f'\_{name}'
	# check if already exists
	if not os.path.exists(baseFolder):
		os.makedirs(baseFolder)
	# algorithm graphs
	filePath = f'{baseFolder}\\_best_k.png'
	# x axis
	len_clusters = len(clusters)
	# y axis
	grad_aic = np.gradient(aic)
	grad_bic = np.gradient(bic)

	# plot
	fig, ( (ax1, ax2), (ax3, ax4) ) = plt.subplots(2, 2)
	# fig size
	fig.set_size_inches(10, 6)
	# aic
	# plot 1
	ax1.set_title('aic')
	ax1.plot( range(2, len_clusters+2), aic)
	# plot 2
	ax2.set_title('aic grad')
	ax2.plot( range(2, len_clusters+2), grad_aic)
	# bic
	# plot 3
	ax3.set_title('bic')
	ax3.plot( range(2, len_clusters+2), bic)
	# plot 4
	ax4.set_title('bic grad')
	ax4.plot( range(2, len_clusters+2), grad_bic)
	# save
	fig.savefig(filePath, dpi=fig.dpi)
	plt.clf()




# tsne graph
def graphTSNE( name, outputFile, original_data, cluster_order, random_state, added_value, apply_pca ):
	# base folder
	baseFolder = outputFile + f'\_{name}'
	# check if already exists
	if not os.path.exists(baseFolder):
		os.makedirs(baseFolder)
	# algorithm graphs
	filePath = f'{baseFolder}\\_clusters.png'
	# apply pca
	if apply_pca:
		pca = PCA(n_components=50)
		original_data = pca.fit_transform(original_data)
	# apply tsne to original dataset
	tsne = mf.TSNE(n_components=2, perplexity=40.0, verbose=2, learning_rate=100.0, early_exaggeration=40.0, n_iter=3000, init='random', method='exact')
	X_np_TSNE = tsne.fit_transform(original_data)
	X_df_TSNE = pd.DataFrame(X_np_TSNE)
	# order dataset in the corresponding clusters
	order_data = {}
	for cluster_num, indexes in cluster_order.items():
		# update indexes, to match the dataset, currently they are based on the excel
		indexes = list(map(lambda x: x-added_value, indexes))
		cluster_idxs = X_df_TSNE.iloc[indexes, : ]
		order_data[cluster_num] = cluster_idxs
	# plot
	fig, ax = plt.subplots(1, 1)
	fig.set_size_inches(10, 6)
	ax.set_title('Clusters')
	colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
	for idx, (cluster_num, data) in enumerate(order_data.items()):
		x = data[0]
		y = data[1]
		ax.plot(x, y, f'.{colors[idx]}')
	# save
	fig.savefig(filePath, dpi=fig.dpi)
	plt.clf()

