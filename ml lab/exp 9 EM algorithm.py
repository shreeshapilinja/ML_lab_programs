'''
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
import sklearn.metrics as sm
'''

from sklearn.datasets import load_iris
import pandas as pd
dataset = load_iris()
x = pd.DataFrame(dataset.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
y = pd.DataFrame(dataset.target, columns=['Targets'])

import numpy as np
colormap = np.array(['red', 'lime', 'black'])    # Create a colormap for visualization

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(14, 7))           # Plot the original data
plt.subplot(1, 3, 1)
plt.scatter(x['petal_length'], x['petal_width'], c=colormap[y['Targets']], s=40)
plt.title('Real')

from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
x_scaled = scaler.fit_transform(x)
x_scaled = pd.DataFrame(x_scaled, columns=x.columns)

from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=3)
gmm.fit(x_scaled)
y_cluster_gmm = gmm.predict(x_scaled)

import sklearn.metrics as sm
print("The accuracy score of EM:", sm.accuracy_score(y['Targets'], y_cluster_gmm))

plt.subplot(1, 3, 3)        # Plot the clusters predicted by GMM
plt.scatter(x['petal_length'], x['petal_width'], c=colormap[y_cluster_gmm], s=40)
plt.title('GMM Clusters')
plt.show()