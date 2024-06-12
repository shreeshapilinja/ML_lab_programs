import pandas as pd
import numpy as np
fulldata = pd.read_csv('enjoysport.csv')
data = np.array(fulldata.iloc[:,:-1])
target = np.array(fulldata.iloc[:,-1])
sp = ['0']*len(data[0])
gn = [["?" for i in range(len(sp))] for i in range(len(sp))]
for i, instance in enumerate(data):
    if target[i].lower() == 'yes':
        for j, attribute in enumerate(instance):
            if sp[j] == '0':
                sp[j] = attribute
            elif sp[j] != attribute:
                sp[j] = '?'
                gn[j][j] = sp[j]
    if target[i].lower() == 'no':
        for j, attribute in enumerate(instance):
            if sp[j] != attribute:
                gn[j][j] = sp[j]

gn = [x for x in gn if not all(val == '?' for val in x)]
print("Specific hypothesis: ",sp)
print("General hypothesis: ",gn)