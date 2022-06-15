import pandas as pd
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, set_link_color_palette
from sklearn.cluster import AgglomerativeClustering, DBSCAN
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Grouping objects by similarity using K-Means
# K-Means clustering using sklearn; use simple 2D dataset; 150 randomly generated points
# Goal is to group samples based on feature similarity
X, y = make_blobs(n_samples=150,
                  n_features=2,
                  centers=3,
                  cluster_std=0.5,
                  shuffle=True,
                  random_state=0)
plt.scatter(X[:, 0],
            X[:, 1],
            c='white',
            marker='o',
            edgecolor='black',
            s=50)
plt.grid()
plt.show()

# K-Means Algorithm
km = KMeans(n_clusters=3,   # Set desired clusters to 3
            init='random',
            n_init=10,  # run algorithm 10 times
            max_iter=300,
            tol=1e-04,
            random_state=0)
y_km = km.fit_predict(X)

# Visualize the clusters
plt.scatter(X[y_km == 0, 0],
            X[y_km == 0, 1],
            s=50, c='lightgreen',
            marker='s', edgecolor='black',
            label='Cluster 1')
plt.scatter(X[y_km == 1, 0],
            X[y_km == 1, 1],
            s=50, c='orange',
            marker='o', edgecolor='black',
            label='Cluster 2')
plt.scatter(X[y_km == 2, 0],
            X[y_km == 2, 1],
            s=50, c='lightblue',
            marker='v', edgecolor='black',
            label='Cluster 3')
plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=250, marker='*',
            c='red', edgecolor='black',
            label='centroids')
plt.legend(scatterpoints=1)
plt.grid()
plt.show()  # See K-Means placed 3 centroids at center of each sphere

# Elbow Method to find optimal number of clusters; plot distortion for different values of k
distortions = []
for i in range(1, 11):
    km = KMeans(n_clusters=i,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=0)
    km.fit(X)
    distortions.append(km.inertia_)
plt.plot(range(1,11), distortions, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Distortion')
plt.show()  # elbow located at k=3; declines most rapidly




# Quantifying the quality of clustering using Silhouette Plots
# Create plot of Silhouette Coefficients for k-means clustering (k=3)
km = KMeans(n_clusters=3,
            init='k-means++',
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0)
y_km = km.fit_predict(X)

cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X,
                                     y_km,
                                     metric='euclidean')
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i) / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper),
             c_silhouette_vals,
             height=1.0,
             edgecolor='none',
             color=color)
    yticks.append((y_ax_lower + y_ax_upper) / 2.)
    y_ax_lower += len(c_silhouette_vals)
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg,
            color='red',
            linestyle='--')
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette Coefficient')
plt.show()  # can scrutinize sizes of different clusters and identify clusters containing outliers

# Bad clustering Silhouette Plot; only uses 2 centroids
km = KMeans(n_clusters=2,
            init='k-means++',
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0)
y_km = km.fit_predict(X)
plt.scatter(X[y_km == 0, 0],
            X[y_km == 0, 1],
            s=50, c='lightgreen',
            edgecolor='black',
            marker='s',
            label='cluster 1')
plt.scatter(X[y_km == 1, 0],
            X[y_km == 1, 1],
            s=50, c='orange',
            edgecolor='black',
            marker='o',
            label='cluster 2')
plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=250, marker='*',
            c='red', label='centroids')
plt.legend()
plt.grid()
plt.show()

# Create silhouette plot to evaluate results
cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X,
                                     y_km,
                                     metric='euclidean')
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(i / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper),
             c_silhouette_vals,
             height=1.0,
             edgecolor='none',
             color=color)
    yticks.append((y_ax_lower + y_ax_upper) / 2)
    y_ax_lower += len(c_silhouette_vals)
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color='red', linestyle='--')
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette Coefficient')
plt.show()  # Silhouettes have visibly different lengths; relatively bad clustering




# Organizing Clusters as a hierarchical tree
# Grouping clusters in bottom-up fashion
# Compute Distance Matrix
np.random.seed(123)
variables = ['X', 'Y', 'Z']
labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']
X = np.random.random_sample([5, 3])*10
df = pd.DataFrame(X, columns=variables, index=labels)
# print(df)

# Performing Hierarchical Clustering on a Distance Matrix
row_dist = pd.DataFrame(squareform(
    pdist(df, metric='euclidean')),
    columns=labels, index=labels)
# print(row_dist)

# Apply linkage agglomeration to clusters using linkage fxn using one of the 2 distance matrix methods:
# Condensed Distance Matrix
# row_clusters = linkage(pdist(df, metric='euclidean'), method='complete')
# Complete Input Sample Matrix
row_clusters = linkage(df.values, method='complete', metric='euclidean')

pd.DataFrame(row_clusters,
             columns=['row label 1',
                      'row label 2',
                      'distance',
                      'no. of items in clust.'],
             index=['cluster %d' %(i+1) for i in
                    range(row_clusters.shape[0])])

# Visualize results using Dendrogram
# make it black pt. 1
# set_link_color_palette(['black'])
row_dendr = dendrogram(row_clusters,
                       labels=labels)   # make dendrogram black pt. 2: add line: color_threshold=np.inf
plt.tight_layout()
plt.ylabel('Euclidean Distance')
plt.show()  # summarizes different clusters that formed during agglomerative hierarchical clustering


# Attaching Dendrograms to a heat map
# Step 1: Create a figure object define axes, rotate dendr 90 degrees counter-clock
fig = plt.figure(figsize=(8, 8), facecolor='white')
axd = fig.add_axes([0.09, 0.1, 0.2, 0.6])
row_dendr = dendrogram(row_clusters, orientation='left')
# for matplotlib < v1.5.1, use orientation='right'

# Step 2: reorder data in df by clustering labels; access through dendr object (leaves key)
df_rowclust = df.iloc[row_dendr['leaves'][::-1]]

# Step 3: construct heat map from reordered df
axm = fig.add_axes([0.23, 0.1, 0.6, 0.6])
cax = axm.matshow(df_rowclust,
                  interpolation='nearest', cmap='hot_r')

# Step 4: modify aesthetics; remove axis ticks; hide axis spines; add color bar; assign feature/sample names to axes
axd.set_xticks([])
axd.set_yticks([])
for i in axd.spines.values():
    i.set_visible(False)
fig.colorbar(cax)
axm.set_xticklabels([''] + list(df_rowclust.columns))
axm.set_yticklabels([''] + list(df_rowclust.index))
plt.show()  # order of rows reflects clustering of samples

# Applying Agglomerative Clustering via Scikit-Learn
ac = AgglomerativeClustering(n_clusters=3,
                             affinity='euclidean',
                             linkage='complete')
labels = ac.fit_predict(X)
print('Cluster Labels: %s' % labels)    # can see that first and fifth sample assigned to cluster 1; consistent w/ dendr

# rerun with n_clusters=2
ac = AgglomerativeClustering(n_clusters=2,
                             affinity='euclidean',
                             linkage='complete')
labels = ac.fit_predict(X)
print('Cluster Labels: %s' % labels)    # pruned clustering hierarchy; 3 was assigned with 0 and 4 as expected



# Section 4: Locating Regions of High Density via DBSCAN
# Illustrate capabilities
X, y = make_moons(n_samples=200,
                  noise=0.05,
                  random_state=0)
plt.scatter(X[:, 0], X[:, 1])
plt.show()  # each half moon has 100 samples each

# See if k-means and complete linkage can identify half-moon shapes as separate clusters
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
km = KMeans(n_clusters=2,
            random_state=0)
y_km = km.fit_predict(X)
ax1.scatter(X[y_km == 0, 0],
            X[y_km == 0, 1],
            c='lightblue',
            edgecolor='black',
            marker='o', s=40,
            label='Cluster 1')
ax1.scatter(X[y_km == 1, 0],
            X[y_km == 1, 1],
            c='red',
            edgecolor='black',
            marker='s', s=40,
            label='Cluster 2')
ax1.set_title('K-Means Clustering')

ac = AgglomerativeClustering(n_clusters=2,
                             affinity='euclidean',
                             linkage='complete')
y_ac = ac.fit_predict(X)
ax2.scatter(X[y_ac == 0, 0],
            X[y_ac == 0, 1],
            c='lightblue',
            edgecolor='black',
            marker='o', s=40,
            label='Cluster 1')
ax2.scatter(X[y_ac == 1, 0],
            X[y_ac == 1, 1],
            c='red',
            edgecolor='black',
            marker='s', s=40,
            label='Cluster 2')
ax2.set_title('Agglomerative Clustering')
plt.legend()
plt.show()  # K-Means is unable to separate 2 clusters; Agglomerative had hard time with shapes

# Implement DBSCAN on dataset to find 2 half-moon shape clusters using density based approach
db = DBSCAN(eps=0.2,
            min_samples=5,
            metric='euclidean')
y_db = db.fit_predict(X)
plt.scatter(X[y_db == 0, 0],
            X[y_db == 0, 1],
            c='lightblue',
            edgecolor='black',
            marker='o', s=40,
            label='Cluster 1')
plt.scatter(X[y_db == 1, 0],
            X[y_db == 1, 1],
            c='red',
            edgecolor='black',
            marker='s', s=40,
            label='Cluster 2')
plt.title('Density-based Spatial Clustering of Applications with Noise (DBSCAN)')
plt.legend()
plt.show()  # highlights strength of DBSCAN: clustering data of arbitrary shapes