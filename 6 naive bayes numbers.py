from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
fulldata = load_iris()
data = fulldata.data
target = fulldata.target
xtrain,xtest,ytrain,ytest = train_test_split(data,target,test_size=0.3,random_state=42)

from sklearn.preprocessing import LabelEncoder
target = LabelEncoder().fit_transform(target)   # automatic text to numbers

from sklearn.naive_bayes import GaussianNB
bclassifier = GaussianNB().fit(xtrain,ytrain)
bpredict = bclassifier.predict(xtest)

from sklearn.metrics import accuracy_score,confusion_matrix
print(confusion_matrix(bpredict,ytest))
print(accuracy_score(bpredict,ytest))