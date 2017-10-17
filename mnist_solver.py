import matplotlib
import gzip
import pandas as pd
# import seaborn as sns
from mnist import MNIST
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

filenames = 'train-images-idx3-ubyte t10k-images-idx3-ubyte train-labels-idx1-ubyte t10k-labels-idx1-ubyte'.split()  # noqa


for i, filename in enumerate(filenames):
    pathin = 'lessons/shared-resources/mnist/' + filename + '.gz'
    pathout = pathin[:-3]
    with gzip.open(pathin) as fin:
        print("Reading file #{}: {}".format(i, pathin))
        with open('lessons/shared-resources/mnist/' + filename, 'wb') as fout:
            print("Writing file #{}: {}".format(i, pathout))
            fout.write(fin.read())


mnistdb = MNIST('lessons/shared-resources/mnist/')

x_train, y_train = mnistdb.load_training()
x_test, y_test = mnistdb.load_testing()

df_train = pd.DataFrame(list(zip(x_train, y_train)), columns=['X', 'y'])
df_test = pd.DataFrame(list(zip(x_test, y_test)), columns=['X', 'y'])

df_train_image = pd.DataFrame(list(df_train.X.values))

pca = PCA(n_components=15).fit(df_train_image)
df_pca = pca.transform(df_train_image)

kmeans = KMeans(n_clusters=10).fit(df_pca)

df_pca['cluster_id'] = kmeans.predict(df_pca[:, :15])
df_pca['digit_id'] = df_train.y
