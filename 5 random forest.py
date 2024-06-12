from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100,max_leaf_nodes=16,n_jobs=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

from sklearn.tree import export_graphviz
export_graphviz(model.estimators_[99], out_file="iris_tree.dot", feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True)

import numpy as np
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])
new_sample_class = model.predict(new_sample)
print(f"Predicted class for the new sample: {iris.target_names[new_sample_class[0]]}")

#dot -Tpng tree.dot -o tree.png