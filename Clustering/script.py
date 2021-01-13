from array import array
import numpy as np
import sys

with open("./outputTrainPredictions.txt", "r") as f:
	images = f.readlines() 

clusters = [[], [], [], [], [], [], [], [], [], []]

# append images per cluster as created by labels from classifier
for i in range(len(images)):
	clusters[int(images[i])].append(i)

# create clusters file as requested
with open("./clustersFile.txt", "a") as f:

	for i in range(len(clusters)):
		f.write("CLUSTER-" + str(i) + "{ size: " + str(len(clusters[i])))
		for j in clusters[i]:
			f.write(" , " + str(j))
		f.write("}\n")

