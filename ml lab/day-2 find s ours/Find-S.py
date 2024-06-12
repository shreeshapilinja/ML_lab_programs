import csv
fulldata = []
with open('enjoysport.csv','r') as csvfile:
	datap = csv.reader(csvfile)
	for row in datap:
		fulldata.append(row)
fulldata.pop(0)
nostrainingexamples = len(fulldata)
nosattributes = len(fulldata[0]) - 1
h = ['0']*nosattributes
h = fulldata[0]
h = h[:-1]
for i in range(nostrainingexamples):
	if fulldata[i][-1] == 'yes':
		fulldata[i].pop()
		for x in fulldata[i]:
			if x!=h[fulldata[i].index(x)]:
				h[fulldata[i].index(x)]='?'
print(h)
