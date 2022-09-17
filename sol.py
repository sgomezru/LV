import os
import math
import numpy as np
import csv
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)

def read_data(fpath):
    with open(fpath) as csvfile:
        readCSV = csv.reader(csvfile, delimiter='\t')
        # NewRotations = []
        # MeasurementQualities = []
        Measurements = []
        MeasurementsByAngle = {}
        for row in readCSV:
            # NewRotation = bool(row[0])
            MeasurementQuality = float(row[1])
            MeasurementAngle = float(row[2])
            if MeasurementQuality > 10 and float(row[3]) < 5000 and float(row[3]) > 500:
                MeasurementDistance = float(row[3])
                Measurements.append((MeasurementAngle, MeasurementDistance))
                MeasurementAngle = round(MeasurementAngle) % 360
                MeasurementsByAngle[MeasurementAngle] = MeasurementsByAngle.get(MeasurementAngle, []) + [MeasurementDistance]
                
    return Measurements, MeasurementsByAngle

def arr_pol2cart(arr):
    '''
    Returns np.array X, with shape N,2. Every
    row is the pair of (x,y) coordinate of each
    point. Before the mapping, the points are
    sorted by angle.
    '''
    X = []
    for phi, rho in sorted(arr, key = lambda x : x[0]):
        X.append(pol2cart(rho, -2*math.pi*phi/360))
    return np.array(X)

def reduce_data(Measurements, method='mean'):
    '''
    @param Measurements: Dictionary, with keys the angles, and values
    the list of distances for those angles.
    @param method: Method to obtain the distance (rho) for each angle
    Available methods:
        - mean
        - median
        - percentile
    '''
    reduced = []
    if method == 'mean':
        for phi in Measurements:
            reduced.append((phi, np.mean(np.array(Measurements[phi]))))
    elif method == 'median':
        for phi in Measurements:
            reduced.append((phi, np.median(np.array(Measurements[phi]))))
    elif method == 'percentile':
        for phi in Measurements:
            _tmp = np.array(Measurements[phi])
            q1, q3 = np.percentile(_tmp, [25,75])
            reduced.append((phi, np.mean(_tmp[np.logical_and(_tmp >= q1, _tmp <= q3)])))
        
    return reduced

def resultsDBScan(X, eps = 500, min_samples = 3):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
        )

        xy = X_complete[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
        )

    plt.title(f"Estimated number of clusters: {n_clusters_} \nNoise points found: {n_noise_}")
    plt.show()

if __name__ == '__main__':
    # data_dir = '/'.join(os.path.abspath(os.path.dirname(__file__)).split('/')[:-1]) + '/data'
    # fpath = os.path.join(data_dir, 'out_startplatz_cut.txt')
    fpath = 'out_startplatz_cut.txt'
    Measurements, MeasurementsByAngle = read_data(fpath)
    X_complete = arr_pol2cart(Measurements)
    X_reduced = arr_pol2cart(reduce_data(MeasurementsByAngle))
    print("Complete data, and reduced data shapes:", X_complete.shape, X_reduced.shape)
    resultsDBScan(X_complete)

