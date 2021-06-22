from collections import Counter
from sklearn import cluster
import json

data = []
file_path = r"../data/test1/data0.txt"
with open(file_path) as f:
    for line in f.readlines()[:27578]:
        data.append(json.loads(f))

lst = []
for x in data:
    lst.append(x[1:])

kmeans = cluster.KMeans(30, lst)
print(Counter(kmeans.labels_))
