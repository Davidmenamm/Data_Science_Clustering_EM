""" Write results to file """

# imports
import os
import pandas as pd



# write to file
def writeFile(name, output_path, ordered_data):
    # base folder
    baseFolder = output_path + f'\_{name}'
    # check if already exists
    if not os.path.exists(baseFolder):
        os.makedirs(baseFolder)
    # file path
    filePath = f'{baseFolder}\\_clusters.csv'
    # print to file
    df = pd.DataFrame.from_dict(data=ordered_data, orient='index').astype(pd.Int64Dtype()) # from dictionary as index
    df.dtypes
    df.to_csv(filePath, sep = ' ', header=False, index = False)



# order data into index per cluster
# added value, adds to each index, to match excel input indexes
def orderData(clusters, added_value = 0):
    # get unique elements (list of all clusters)
    cluster_idx = set(clusters)
    # num of clusters
    num_clusters = len(cluster_idx)
    # dictionary of idx's per cluster
    cluster_dict = {key: list() for key in cluster_idx}
    # iterate elements
    for idx, clusterNum in enumerate(clusters):
        # add to each idx
        idx_actualized = idx + added_value
        # add idx to dictionary
        cluster_dict[clusterNum].append(idx_actualized)
    print('cluster_dict')
    for k,v in cluster_dict.items():
        print(f'{k} -> {v}\n')
    # return
    return cluster_dict



