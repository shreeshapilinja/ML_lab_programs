import numpy as np
import pandas as pd
data = pd.read_csv('enjoysport.csv')
concept = np.array(data)[:,0:-1]
target = np.array(data)[:,-1:]

def finds(concept,target):
	sp = concept[0]
	for i,val in enumerate(concept):
		if target[i]=='yes':
			for x in range(len(concept[0])):
				if val[x]!=sp[x]:
					sp[x]='?'
	return sp

def candidateelimination(concept,target):
	sp = concept[0]
	gn = [['?' for _ in range(len(sp))] for _ in range(len(sp))]
	for i,val in enumerate(concept):
		if target[i] =='yes':
			for x in range(len(concept[0])):
				if val[x]!=sp[x]:
					sp[x]='?'
					gn[x][x]='?'
		if target[i] == 'no':
			for x in range(len(concept[0])):
				if val[x]!=sp[x]:
					gn[x][x] = sp[x]
				else:
					gn[x][x] = '?'
	indices = [i for i, val in enumerate(gn) if val == ['?', '?', '?', '?', '?', '?']]
	for i in indices:
		gn.remove(['?', '?', '?', '?', '?', '?']) 
	return sp,gn

print(finds(concept,target))
print('\n')

(sp,gn) = candidateelimination(concept,target)
print(sp)
print(gn)

