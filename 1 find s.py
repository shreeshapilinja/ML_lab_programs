import pandas as pd
import numpy as np
fulldata = pd.read_csv('enjoysport.csv')
data = np.array(fulldata.iloc[:,:-1])
target = np.array(fulldata.iloc[:,-1])
sp = ['0']*len(data[0])
for i, instance in enumerate(data):
    if target[i].lower() == 'yes':
        for j, attribute in enumerate(instance):
            if sp[j] == '0':
                sp[j] = attribute
            elif sp[j] != attribute:
                sp[j] = '?'
print("Specific hypothesis: ",sp)