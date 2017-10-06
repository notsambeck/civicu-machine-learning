# import seaborn as sns
from matplotlib import pyplot as plt
# import mpld3
from mpl_toolkits.mplot3d import Axes3D  # noqa
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# mpld3.enable_notebook()
np = pd.np

df = pd.read_csv('shared-resources/pointcloud.csv.gz', header=0, index_col=0)
# df.to_csv('../shared-resources/pointcloud.csv.gz', compression='gzip')
df = df.sample(1000).copy()
df.head(3)

kmeans = KMeans(init='k-means++', n_clusters=5)
kmeans = kmeans.fit(df)
df['cluster_id'] = kmeans.predict(df)
colors = np.array(list('rgbkcmyrgbkcmy'))[df.cluster_id.values]
# df.plot.scatter(x='x', y='y', c=colors)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax = ax.scatter(df.x, df.y, df.z, c=colors)

tsne = TSNE()
# t-distributed stochastic non-linear embedding
# similar to randomized PCA but with warping
# t-student distribution is similar to normal but with more outliers
tsne.fit(df[list('xyz')])

df_tsne = pd.DataFrame(tsne.embedding_, columns=list('xy'))

tsne_kmeans = KMeans(init='k-means++', n_clusters=7)
tsne_kmeans = tsne_kmeans.fit(df_tsne)
df_tsne['cluster_id'] = tsne_kmeans.predict(df_tsne)
colors = np.array(list('rgbkcmy'))[df_tsne.cluster_id.values]
df_tsne.plot.scatter(x='x', y='y', c=colors)

'''
fig2 = plt.figure()
ax = fig2.add_subplot(111, projection='3d')
ax = ax.scatter(df_tsne.x, df_tsne.y, df_tsne.z, c=colors)
'''

plt.show()
