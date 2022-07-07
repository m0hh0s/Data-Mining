import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import matplotlib
from matplotlib import colors


# loading the dataset
df = pd.read_csv("iris.data", names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target'])
# missing values
df = df.dropna(how='any')
# label encoding
le = LabelEncoder()
le.fit(["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
df['target'] = le.transform(df['target'])
# normalization
scaler = StandardScaler()
scaler.fit(df.get(['sepal_length', 'sepal_width', 'petal_length', 'petal_width']))
scaled_data = scaler.transform(df.get(['sepal_length', 'sepal_width', 'petal_length', 'petal_width']))
df['sepal_length'] = scaled_data[:, 0]
df['sepal_width'] = scaled_data[:, 1]
df['petal_length'] = scaled_data[:, 2]
df['petal_width'] = scaled_data[:, 3]
# PCA
pca = PCA(n_components=4)
pca.fit(df.get(['sepal_length', 'sepal_width', 'petal_length', 'petal_width']))
pca_out = pca.transform(df.get(['sepal_length', 'sepal_width', 'petal_length', 'petal_width']))
# visualization
color_indices = df.get('target')
color = ['yellow', 'green', 'orange']
colormap = matplotlib.colors.ListedColormap(color)
# plt.scatter(pca_out[:, 0], pca_out[:, 1], c=color_indices, cmap=colormap)
plt.boxplot(df.get('petal_width'))
plt.show()




