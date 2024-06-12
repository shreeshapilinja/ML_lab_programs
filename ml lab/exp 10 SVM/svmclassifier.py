import pandas as pd
data = pd.read_csv('SocialNetworkAds.csv')
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.3,random_state=42)

from sklearn.preprocessing import StandardScaler
s = StandardScaler()
Xtrain = s.fit_transform(Xtrain)
Xtest = s.transform(Xtest)

from sklearn.svm import SVC
sc = SVC(kernel='rbf',random_state=0).fit(Xtrain,ytrain)
ypred = sc.predict(Xtest)

from sklearn.metrics import accuracy_score,confusion_matrix
print(accuracy_score(ytest,ypred))
print(confusion_matrix(ytest,ypred))
