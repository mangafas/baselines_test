import numpy as np
import pandas as pd
import os, copy
from sklearn.cluster import KMeans
from sklearn import metrics
from collections import Counter
import matplotlib.pyplot as plt

""" eval
Tested with python3.
"""


def computeBicAic(model=None, likelihood=0.0):
    datapoints = len(model.labels_)
    parameters = len(model.get_params())
    #print(model.get_params())
    BIC = (np.log(datapoints) * parameters) - (2.0 * np.log(likelihood))

    AIC = (2.0 * parameters) - (2.0 * np.log(likelihood))
    return BIC, AIC


def getAffordanceLabels(clustering_data=None, clustering_ids=None, groundtruth_ids=None):
    affordances_ = []
    i = 0
    while i < len(clustering_ids):
        data = clustering_ids.loc[i]
        #print(groundtruth_ids)
        vid = data['video']
        obj = int(data['object'])
        not_detected = False
        #cstr = str(groundtruth_ids.loc[i]['object'])
        #print(str(groundtruth_ids['object']))
        #nv = int(groundtruth_ids['video']) == vid
        nv = groundtruth_ids['video'] == vid
        #print(groundtruth_ids)
        #print("data,obj,cstr,nv")
        #print(data)
        #print(data)
        #print(obj)
        #print(cstr)
        #print(nv)
        try:
            # no = groundtruth_ids['Detected_objects'] == obj
            #no = int(groundtruth_ids['object']) == obj
            no = groundtruth_ids['object'] == obj

            aff = groundtruth_ids[nv & no]['Affordance_labels'].values
        except ValueError:
            not_detected = True

        if not_detected:  # don't use that information in the clustering.
            clustering_data = np.delete(clustering_data, (i), axis=0)
            clustering_data = np.delete(clustering_data, (i), axis=1)
            #print("not detected at " + str(i))
        elif len(aff) == 0:  # if this object has no correlation with a ground truth object
            affordances_.append('NA')
            #print("length is zero at " + str(i))
            i += 1
        else:
            #print("detected mate")
            affl = aff.tolist()
            affl.sort()
            affordances_.append('_'.join(affl))
            i += 1

    df = pd.DataFrame({'Affordance_labels': affordances_})
    labels = df['Affordance_labels'].values
    return labels, clustering_data


def evaluate_clustering(clusters, labels):
    #print("labels start")
    #print(labels)
    #print("labels end")
    predictions = copy.deepcopy(clusters)
    labels = denoise_FP_labels(clusters, labels)
    labels = denoise_double_labels(clusters, labels)

    # Create cluster groups.
    cnumber = max(np.unique(predictions))
    clusters = [[] for i in range(cnumber + 1)]
    for i in range(len(predictions)):
        clusters[predictions[i]].append(i)

    # Check cluster labels
    for c in range(cnumber + 1):
        l_ = [labels[i] for i in range(len(predictions)) if predictions[i] == c]
        idx = [i for i in range(len(predictions)) if predictions[i] == c]
        unique, counts = list(np.unique(l_, return_counts=True))
        counts = list(counts)
        max_pos = counts.index(max(counts))

    # Compute metrics for cluster evaluation.
    vmeasure = metrics.cluster.v_measure_score(labels, predictions)
    #print("labels - predictions ")
    #print(labels, predictions)
    nmi_score = metrics.cluster.normalized_mutual_info_score(labels, predictions)
    homoscore = metrics.cluster.homogeneity_score(labels, predictions)
    compscore = metrics.cluster.completeness_score(labels, predictions)
    return vmeasure, nmi_score, homoscore, compscore


def denoise_FP_labels(clusters, labels):
    num_clusters = max(clusters) + 1
    new_label = 0
    for c in range(num_clusters):
        datapoints = [(labels[i], i) for i in range(len(labels)) if clusters[i] == c]
        datapoints_labels = list(map(lambda x: x[0], datapoints))
        datapoints_idx = list(map(lambda x: x[1], datapoints))
        counts = Counter(datapoints_labels)
        sorted_counts = sorted(counts.items(), key=lambda x: x[1])

        # find dominant label
        i = 0
        dominant = sorted_counts[0]
        while i < len(counts):
            if dominant[0] == 'NA':
                i += 1
                try:
                    dominant = sorted_counts[i]
                except IndexError:
                    dominant = ('label' + str(new_label), 0)
                    new_label += 1

            else:
                break
        dominant_label = dominant[0]

        # change noisy labels of that cluster with the dominant one
        for i in range(len(datapoints_idx)):
            idx = datapoints_idx[i]
            label = datapoints_labels[i]
            if label == 'NA':
                labels[idx] = dominant_label

    return labels


def denoise_double_labels(clusters, labels):
    num_clusters = max(clusters) + 1
    for c in range(num_clusters):
        datapoints = [(labels[i], i) for i in range(len(labels)) if clusters[i] == c]
        datapoints_labels = list(map(lambda x: x[0], datapoints))
        datapoints_idx = list(map(lambda x: x[1], datapoints))
        # split the double labels
        split_labels = [dp.split('_') for dp in datapoints_labels]
        split_labels = sum(split_labels, [])
        counts = Counter(split_labels)
        sorted_counts = sorted(counts.items(), key=lambda x: x[1])

        # find dominant label
        i = 0
        dominant = sorted_counts[0]
        while i < len(counts):
            if ('newlabel' in dominant[0]):
                i += 1
                try:
                    dominant = sorted_counts[i]
                except IndexError:
                    dominant = sorted_counts[0]
                    break
            else:
                break
        dominant_label = dominant[0]

        # change noisy labels of that cluster with the dominant one
        for i in range(len(datapoints_idx)):
            idx = datapoints_idx[i]
            label = datapoints_labels[i]
            if ((dominant_label in label) and ('_' in label)) or ('newlabel' in label):
                labels[idx] = dominant_label

    return labels


# Initialize paths and variables. #TODO
find_bic_aic = False
fold_number = 4

fold = 'test'  # ['trainval', 'test', 'test_wnp', 'test_load']
n_clusters = 32
max_n_clusters = 35
clusteringFile = r'C:\Users\manga\Desktop\table.npy'  # Add path to .NPY file with clustering data
data_file = r'C:\Users\manga\Desktop\Datasets\CAD-120\converter.csv'  # Add path to .CSV identification data file
GT_PATH = r'C:\Users\manga\Downloads\data\CAD-120\groundtruth\\'  # Add path to /GT2_foldX.csv files with groundtruth data

# Load clustering data.
print('Loading the clustering data...', end='')
X = np.load(clusteringFile)
print('Done.')

# Load groundtruth data.
multipleGT = True if 'train' in fold else False
if multipleGT:  # for training data
    gtData = []
    for i in [j for j in range(1, 6) if j != fold_number]:
        gt_file = GT_PATH + 'GT2_fold_' + fold + str(i) + '.csv'
        df_gt = pd.read_csv(gt_file)
        df_gt.rename(columns={'Unnamed: 0': 'idx'}, inplace=True)
        gtData.append(df_gt)

    df_gt = pd.concat(gtData, ignore_index=True)
else:  # for testing data
    gt_file = GT_PATH + 'GT2_fold_' + fold + '.csv'
    df_gt = pd.read_csv(gt_file)
    df_gt.rename(columns={'Unnamed: 0': 'idx'}, inplace=True)

# Load identification data.
df_data = pd.read_csv(data_file)
df_data.rename(columns={'Unnamed: 0': 'idx'}, inplace=True)

# :: Keep only the important path for the video identification. ::
# Delete the '/usr/share/Datasets/CAD-120/images/' part from the saved video names.
# [ Comment this out if not needed. #TODO]
# df_data['Video'] = pd.np.where(df_data.Video.str.contains('Datasets/CAD-120/images/'),
#                                df_data.Video.str.replace('Datasets/CAD-120/images/',''),
#                                df_data.Video)
# df_data['Video'] = pd.np.where(df_data.Video.str.contains('Datasets/Watch-n-Patch/images/'),
#                                df_data.Video.str.replace('Datasets/Watch-n-Patch/images/',''),
#                                df_data.Video)
# df_data['Video'] = pd.np.where(df_data.Video.str.contains('Datasets/LOAD/images/'),
#                                df_data.Video.str.replace('Datasets/LOAD/images/',''),
#                                df_data.Video)
#

# Get groundtruth afffordance labels.
print('Get affordance labels...', end='')
labels, X = getAffordanceLabels(clustering_data=X,
                                clustering_ids=df_data,
                                groundtruth_ids=df_gt)
print('Done.')
mbic = maic = 100000000
# Clustering and Evaluation
if find_bic_aic:
    Bic, Aic = [], []
    n_clusters_values = list(range(1, max_n_clusters))
    for n_clusters in n_clusters_values:
        for rs in range(15):
            kmeans = KMeans(centroids=n_clusters)  # TODO
            clusters = kmeans.fit_predict(X)
            vmeasure, _, _, _ = evaluate_clustering(clusters, labels)
            bic, aic = computeBicAic(model=kmeans, likelihood=vmeasure)
            Bic.append(bic)
            Aic.append(aic)
            print("number of clusters: " + str(n_clusters))
            print(bic, aic)
            if bic < mbic:
                mbic = bic
                mbclus = n_clusters
                mrs = rs
            if aic < maic:
                maic = aic
                maclus = n_clusters
    # plot BIC and AIC values
    print(mbclus, mbic, mrs)
    print(maclus, maic)
    plt.plot(Bic, 'bo')
    plt.plot(Aic, 'r+')
    plt.show()

else:
    vt = ht = ct = 0
    for x in range(5):
        kmeans = KMeans(n_clusters=n_clusters)  # TODO
        clusters = kmeans.fit_predict(X)

        vmeasure, _, homoscore, compscore = evaluate_clustering(clusters, labels)
        print("vmeasure, homoscore, compscore")
        print(vmeasure, homoscore, compscore)
        vt += vmeasure
        ht += homoscore
        ct += compscore
    print(round(vt/5,4),round(ht/5,4),round(ct/5,4))


