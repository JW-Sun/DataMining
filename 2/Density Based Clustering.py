from sklearn import preprocessing
import pandas as pd
import math
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import DBSCAN
import networkx as nx
import numpy as np
from sklearn.datasets import load_iris
from sklearn.base import BaseEstimator, ClusterMixin
import networkx as nx


iris = datasets.load_iris()
mg=pd.read_csv('iris.txt',sep=',',header=None,names=(['1','2','3','4','5']))
del mg['5']
data=mg.values;
data1=mg.values
#print(data)
class Denclue(BaseEstimator, ClusterMixin):
    def __init__(self, h=None, eps=1e-5, min_density=0., metric='euclidean'):
        self.h = h
        self.eps = eps
        self.min_density = min_density
        self.metric = metric

    def gauss_kernel(self, x, y, h, d):
        kernel = np.exp(-(np.linalg.norm(x - y) / h) ** 2. / 2.) / ((2. * np.pi) ** (d / 2))
        return kernel

    def iteration_gauss_kernel(self, x_l0, X, h=0.1):
        n, d = np.shape(X)
        denomation = 0.
        _x_t1 = np.zeros((1, d))
        for j in range(n):
            kernel = self.gauss_kernel(x_l0, X[j], h, d)
            kernel = kernel / (h ** d)
            denomation += kernel
            _x_t1 = _x_t1 + (kernel * X[j])
        _x_t1 = _x_t1 / denomation
        density = denomation / n
        return [_x_t1, density]

    def find_attractor(self, x_t, X, h=0.1, eps=1e-5):
        now_eps = 100.
        pre = 0.
        _x_t1 = np.copy(x_t)

        radius_new = 0.
        radius_old = 0.
        radius_twiceold = 0.
        while now_eps >= eps:
            radius_thriceold = radius_twiceold
            radius_twiceold = radius_old
            radius_old = radius_new
            _x_t0 = np.copy(_x_t1)
            _x_t1, density = self.iteration_gauss_kernel(_x_t0, X, h=h)
            now_eps = density - pre
            pre = density
            radius_new = np.linalg.norm(_x_t1 - _x_t0)
            radius = (radius_thriceold + radius_twiceold + radius_old + radius_new)
        return [_x_t1, density, radius]

    def denclue_algo(self, X):
        row, col = np.shape(X)
        self.n_samples = row
        self.n_features = col
        density_attractors = np.zeros((self.n_samples, self.n_features))
        radii = np.zeros((self.n_samples, 1))
        density = np.zeros((self.n_samples, 1))

        for i in range(self.n_samples):
            density_attractors[i], density[i], radii[i] = \
                self.find_attractor(X[i], X, h=self.h, eps=self.eps)

        cluster_info = {}
        num_clusters = 0
        cluster_info[num_clusters] = {'nodes': [0]}
        g_clu = nx.Graph()
        for j1 in range(self.n_samples):
            g_clu.add_node(j1, attr_dict={'attractor': density_attractors[j1], 'radius': radii[j1],
                                               'density': density[j1]})


        for j1 in range(self.n_samples):
            for j2 in (x for x in range(self.n_samples) if x != j1):
                if g_clu.has_edge(j1, j2):
                    continue
                diff = np.linalg.norm(
                    g_clu.node[j1]['attr_dict']['attractor'] - g_clu.node[j2]['attr_dict']['attractor'])
                if diff <= (g_clu.node[j1]['attr_dict']['radius'] + g_clu.node[j2]['attr_dict']['radius']):
                    g_clu.add_edge(j1, j2)

        clusters = list(nx.connected_component_subgraphs(g_clu))
        num_clusters = 0

        labels = []
        for clust in clusters:

            max_location = max(clust, key=lambda x: clust.node[x]['attr_dict']['density'])
            max_density = clust.node[max_location]['attr_dict']['density']
            # populate cluster_info dict
            cluster_info[num_clusters] = {'nodes': clust.nodes(),
                                          'density': max_density}

            if max_density >= self.min_density:
                # density attractor points
                labels.append(max_location)
            else:
                # noise points
                labels.append(-1)
            num_clusters += 1

        self.clust_info_ = cluster_info
        self.labels_ = labels
        return self

denclue = Denclue(h=0.1, eps=0.0001, min_density=2.85)
iris = load_iris()
data = iris.data
results = denclue.denclue_algo(data1)
num = 0

cluster_info = results.clust_info_
lables = results.labels_

for i in range(lables.__len__()):
    if lables[i] != -1:
        str_cluster_info = str(cluster_info[i]['nodes'])
        cluster = str_cluster_info[1: str_cluster_info.__len__() - 1]
        _cluster = cluster.split(',')
        cluster_arr = list(map(int, _cluster))
        print(str(num + 1) + ':' + str_cluster_info)
        print('簇的大小：' + str(cluster_arr.__len__()))
        print('密度吸引子：' + str(lables[i]))
        num += 1
def DENCLUE(D, eps=0.3, min_samples=10):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(D)
    coreSampleMask = np.zeros_like(db.labels_, dtype=bool)
    coreSampleMask[db.core_sample_indices_] = True
    clusterLabels = iris.target
    uniqueClusterLabels = set(clusterLabels)
    colors = ['red', 'Cyan', 'blue', 'yellow', 'black']
    markers = ['x', '*', '+', 'o', '8', 'H', '3']
    for i, cluster in enumerate(uniqueClusterLabels):
        clusterIndex = (clusterLabels == cluster)
        coreSamples = D[clusterIndex & coreSampleMask]
        plt.scatter(coreSamples[:, 0] + coreSamples[:, 1], coreSamples[:, 2] + coreSamples[:, 3], c=colors[i],
                    marker=markers[i], s=80)
        noiseSamples = D[clusterIndex & ~coreSampleMask]
        plt.scatter(noiseSamples[:, 0] + noiseSamples[:, 1], noiseSamples[:, 2] + noiseSamples[:, 3], c=colors[i],
                    marker=markers[i], s=26)
    plt.show()


DENCLUE(data, 0.0001, 0.03)
def get_density(self, x, X, y=None, sample_weight=None):
    superweight = 0.
    n_samples = X.shape[0]
    n_features = X.shape[1]
    if sample_weight is None:
        sample_weight = np.ones((n_samples, 1))
    else:
        sample_weight = sample_weight
    for y in range(n_samples):
        kernel = kernelize(x, X[y], h=self.h, degree=n_features)
        kernel = kernel * sample_weight[y] / (self.h ** n_features)
        superweight = superweight + kernel
    density = superweight / np.sum(sample_weight)
    return density


def kernelize(x, y, h, degree):
    kernel = np.exp(-(np.linalg.norm(x - y) / h) ** 2. / 2.) / ((2. * np.pi) ** (degree / 2))
    return kernel