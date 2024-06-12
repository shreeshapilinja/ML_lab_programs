from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
rnd_clf = RandomForestClassifier(n_estimators=100, max_leaf_nodes=16, n_jobs=-1)
iris=load_iris()
x=iris.data[:,:]
y=iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

rnd_clf.fit(x_train, y_train)
y_pred = rnd_clf.predict(x_test)
print(accuracy_score(y_test,y_pred))


from sklearn.tree import export_graphviz

export_graphviz(rnd_clf.estimators_[0],out_file="iris_tree.dot",feature_names=iris.feature_names,class_names=iris.target_names,rounded=True,filled=True)
