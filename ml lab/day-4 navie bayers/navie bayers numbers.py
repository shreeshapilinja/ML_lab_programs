from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
data=pd.read_csv("Iris.csv")
x=np.array(data.iloc[:,1:-1])
y=np.array(data.iloc[:,-1])
for i,h in enumerate(y):
	if y[i]=='Iris-setosa':
		y[i]='0'
	elif y[i]=="Iris-versicolor":
		y[i]='1'
	else:
		y[i]='2'

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=42)
classifier=GaussianNB()
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)

pc=0
nc=0
for i,h in enumerate(y_pred):
	if y_test[i]==y_pred[i]:
		pc+=1
	else:
		nc+=1
tot=len(y_pred)
pc=(pc/tot)*100
nc=(nc/tot)*100
if pc>nc:
	print("Probability=",pc," which is positive")
else:
	print("Probability=",nc," which is negative")
	
	
	
	
	
	
