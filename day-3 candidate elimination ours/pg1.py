import csv
data = []
datacopy = []
with open('enjoysport.csv','r') as csvfile:
	csvdata = csv.reader(csvfile)
	for i in csvdata:
		data.append(i)
		datacopy.append(i)
datacopy.pop(0)

target = []
for i in datacopy:
	target.append(i[-1])


for i in datacopy:
	i.pop()
concepts = datacopy

nosattri = len(concepts[0])

sp = ['0']*nosattri
gn = [['?']*nosattri for i in range(nosattri)]

sp = concepts[0]

for i,j in enumerate(concepts):
	if target[i] == 'Yes' or target[i] == 'yes':
		for x in range(nosattri):
			if j[x]!= sp[x]:                    
				sp[x] ='?'
				gn[x][x] ='?'

	if target[i] == "No" or target[i] == "no":            
		    for x in range(nosattri): 
		        if j[x]!= sp[x]:                    
		            gn[x][x] = sp[x]           
		        else:                    
		            gn[x][x] = '?'

print('specific hypothesis is =',sp)
print('\n')
indices = [i for i, val in enumerate(gn) if val == ['?', '?', '?', '?', '?', '?']]    
for i in indices:   
        gn.remove(['?', '?', '?', '?', '?', '?']) 

print('general hypothesis is =',gn)
