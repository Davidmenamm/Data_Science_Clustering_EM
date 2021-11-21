""" Manage All the Program """

# imports
import timeit
from classification import classify
from os import listdir, path
from preprocessing import getTopInfo, readDataset, formatTable
from graph import drawTable, graphCurve, bestK, graphTSNE
from writeFile import orderData, writeFile

# input paths
baseInPath = r'data\input'

# output paths
baseOutPath = r'data\output'


# Coordinator
def coordinate():
    # time init
    tic = timeit.default_timer()
    # paths and names for hypothesis
    fileNames = [f.replace('.csv', '') for f in listdir(
        baseInPath) if path.isfile(path.join(baseInPath, f))]
    filePaths = [path.join(baseInPath, f) for f in listdir(
        baseInPath) if path.isfile(path.join(baseInPath, f))]
    fileInformation = zip(fileNames, filePaths)
    # loop hypothesis
    for info in fileInformation:
        # read dataset and apply balancing
        df = readDataset(info[1])
        # run classification
        k_range = range(2, 10+1)
        random_state = 100
        num_init = 100
        clusters_list, aic, bic = classify( df, k_range, num_init, random_state )
        # find best k, in graph
        bestK( info[0], baseOutPath, clusters_list, aic, bic,  )
        best_k_value = 5 # after looking at the graph
        # print clusters in file
        best_results = clusters_list[best_k_value-2] # -2 because it starts in two, and since array indices start in 0
        print(best_results)
        added_value = 2
        ordered_dict = orderData(best_results, added_value)
        writeFile( info[0], baseOutPath, ordered_dict )
        # print tsne plot
        graphTSNE( info[0], baseOutPath, df, ordered_dict, random_state, added_value, False)
    # time end
    toc = timeit.default_timer()
    elapsed = toc-tic
    print(f'Time elapsed is aproximately {elapsed} seconds o {elapsed/60} minutes')
