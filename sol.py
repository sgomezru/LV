import os
import math
import numpy as np
import cv2 as cv
import csv
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import DBSCAN
import sys

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

def distNoiseRemoval(X, threshold = 500):
    validIndices = []
    invalidIndices = []
    for idx in range(len(X) - 1):
        if (np.linalg.norm(X[idx] - X[idx - 1]) > threshold) and (np.linalg.norm(X[idx] - X[idx + 1]) > threshold):
            invalidIndices.append(idx)
        else:
            validIndices.append(idx)
    # Only for last point(Could have use % operator before as well)
    idx = len(X)- 1
    if (np.linalg.norm(X[-1] - X[-2]) > threshold) and (np.linalg.norm(X[-1] - X[0]) > threshold):
        invalidIndices.append(idx)
    else:
        validIndices.append(idx)
    return X[validIndices], validIndices, invalidIndices

def plotCartesian(X, markerType='markers', title=''):
    fig = go.Figure(data=go.Scatter(x=X[:,0], y=X[:,1], mode=markerType))
    fig.update_layout(title_text=title, yaxis=dict(scaleanchor="x", scaleratio=1))
    fig.show()

##################### Split and Merge Algorithm (IEPF) part ##################################

def Polar2Cartesian(r, alpha):
    return np.transpose(np.array([np.cos(alpha)*r, np.sin(alpha)*r]))

def Cartesian2Polar(x, y):
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return r, phi

### Note: Commented the means, since different lines would be created in contrast to originally scaled data.
### my reasoning about it may be wrong though, but I guess it's not needed for now anyway.
def GetPolar(X,Y):
    # X = X-np.mean(X)
    # Y = Y-np.mean(Y)
    # fit line through the first and last point (X and Y contains 2 points, start and end of the line)
    k,n = np.polyfit(X,Y,1)
    alpha = math.atan(-1/k) # in radians
    ro = n/(math.sin(alpha)-k*math.cos(alpha))
    return ro,alpha

def CheckPolar(ro,alpha):
    if ro < 0:
        alpha = alpha + math.pi
        if alpha > math.pi:
            alpha = alpha-2*math.pi
        ro = -ro
    return ro,alpha

### Note: Weird here, taking abs of a norm (?)
def getDistance(P,Ps,Pe): # point to line distance, where the line is given with points Ps and Pe
    if np.all(np.equal(Ps,Pe)):
        return np.linalg.norm(P-Ps)
    return np.divide(np.abs(np.linalg.norm(np.cross(Pe-Ps,Ps-P))),np.linalg.norm(Pe-Ps))

def GetMostDistant(P):
    dmax = 0
    index = -1
    for i in range(1,P.shape[0]):
        d = getDistance(P[i,:],P[0,:],P[-1,:])
        if (d > dmax):
            index = i
            dmax = d
    return dmax,index

def SplitAndMerge(P,threshold):
    d,ind = GetMostDistant(P)
    if (d>threshold):
        P1 = SplitAndMerge(P[:ind+1,:],threshold) # split and merge left array
        P2 = SplitAndMerge(P[ind:,:],threshold) # split and merge right array
        points = np.vstack((P1[:-1,:],P2))
    else:
        points = np.vstack((P[0,:],P[-1,:]))
    return points

def callback(data, threshold = 50):
    points = SplitAndMerge(data, threshold)
    rs, alphas = [], []
    for i in range(points.shape[0]-1):
        r, alpha = GetPolar(points[i:i+2,0], points[i:i+2,1])
        r, alpha = CheckPolar(r, alpha)
        rs.append(r)
        alphas.append(alpha)
        
    print(f"Original number of points: {data.shape}")
    print(f"After split and merge with threshold = {threshold}: {points.shape}")

    return points, np.array(rs), np.array(alphas)

############################# Split & Merge finish ##################################################


################# As an image part ##################################

### Note: I could have also taken directly an exported image from plotly/matplotlib.pyplot
### which I believe uses some kind of interpolation to plot stuff.
def data2Image(X, scaling = 0.01):
    # Just scaling, and making data integer, to map into image pixels.
    meta = {'scaling': scaling}
    pads = 5
    tmp = X.copy()
    tmp *= scaling
    tmp = np.around(tmp)
    tmp = tmp.astype(int)
    # Moving the data into the first quadrant basically
    min_x, max_x = np.min(tmp[:, 0]), np.max(tmp[:, 0])
    min_y, max_y = np.min(tmp[:, 1]), np.max(tmp[:, 1])
    meta['width'], meta['height'] = max_x + 1 - min_x, max_y + 1 - min_y
    meta['offset_x'], meta['offset_y'] = min_x, min_y
    meta['pad'] = pads
    tmp[:, 0] -= min_x
    tmp[:, 1] -= min_y
    img = np.full((meta['width'], meta['height']), 0, dtype=np.uint8)
    # Might be a better numpy advanced indexing method(?) 
    for x, y in tmp:
        img[x,y] = 255
    for i in range(pads):
        img = np.pad(img, (1,1), constant_values = (0,0))
    return img, meta, tmp

def showImg(img, binary=True):
    if binary:
        fig = px.imshow(img, binary_string=True)
    else:
        fig = px.imshow(img)
    fig.show()

def lineSegmentDetector(img_path):
    img = cv.imread(img_path, 0)
    dil = cv.dilate(img, kernel=np.ones((3,3), np.uint8), iterations = 1)
    lsd = cv.createLineSegmentDetector(0)
    lines = lsd.detect(dil)[0] #Position 0 of the returned tuple are the detected lines
    drawn_img = lsd.drawSegments(img,lines)
    showImg(drawn_img, False)
    return drawn_img

def goodFeaturesToTrack(img_path, ncorners = 10):
    img = cv.imread(img_path, 0)
    corners = cv.goodFeaturesToTrack(img,ncorners,0.01,10)
    corners = np.int0(corners)
    for i in corners:
        x,y = i.ravel()
        cv.circle(img,(x,y),3,255,-1)
    showImg(img)
    return img, corners

if __name__ == '__main__':
    # data_dir = '/'.join(os.path.abspath(os.path.dirname(__file__)).split('/')[:-1]) + '/data'
    # fpath = os.path.join(data_dir, 'out_startplatz_cut.txt')
    argDBSCAN = False
    argNeighbor = True
    argSplitMerge = False
    argImage = True
    argPlot = False
    fpath = 'out_startplatz_cut.txt'
    Measurements, MeasurementsByAngle = read_data(fpath)
    X_complete = arr_pol2cart(Measurements)
    X_reduced = arr_pol2cart(reduce_data(MeasurementsByAngle))
    print("Complete data, and reduced data shapes:", X_complete.shape, X_reduced.shape)
    if argDBSCAN:
        resultsDBScan(X_complete)
    if argNeighbor:
        validData, _, _ = distNoiseRemoval(X_complete)
        validReducedData, _, _ = distNoiseRemoval(X_reduced)
        print(f"Original num points: {len(X_complete)}\t Num points after noise: {len(validData)}")
        if argPlot:
            plotCartesian(validData, title='Graph as points with all data after noise removal')
            plotCartesian(validData, 'lines', title='Graph as lines with all data after noise removal')
            plotCartesian(validReducedData, 'lines', title='Graph as lines with all data after noise removal')
    if argSplitMerge:
        points, _, _ = callback(X_complete)
        if argPlot:
            plotCartesian(points, 'markers', title=f'With whole original data: After split & merge with threshold = 50')
        if argNeighbor:
            points, _, _ = callback(validReducedData)
            if argPlot:
                plotCartesian(points, 'lines', title=f'With reduced data & "noise removal": After split & merge with threshold = 50')
                thresh = 50, 100, 200
                for t in thresh:
                    points, _, _ = callback(validData, t)
                    plotCartesian(points, 'lines', title=f'With whole data after "noise removal": After split & merge with threshold = {t}')
    if argImage:
        img, metadata, datatmp = data2Image(X_complete)
        cv.imwrite("original.jpg", img)
        showImg(img)
        lsdImg = lineSegmentDetector('original.jpg')
        cv.imwrite('lsdImage.jpg', lsdImg)
        gfttImg, cornerPoints = goodFeaturesToTrack('original.jpg', 10)
        cv.imwrite('gfttImage.jpg', gfttImg)

