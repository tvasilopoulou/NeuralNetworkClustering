from scipy.stats import wasserstein_distance
import numpy as np
from pulp import *
import sys

def condition(x, number): return x > number

def blockshaped(arr, nrows, ncols):
	"""
	Return an array of shape (n, nrows, ncols) where
	n * nrows * ncols = arr.size

	If arr is a 2D array, the returned array should look like n subblocks with
	each subblock preserving the "physical" layout of arr.
	"""
	h, w = arr.shape
	assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
	assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
	return (arr.reshape(h//nrows, nrows, -1, ncols).swapaxes(1,2).reshape(-1, nrows, ncols))


def EMD(feature1, feature2, w1, w2):
    os.environ['PATH'] += os.pathsep + '/usr/local/bin'

    H = feature1.shape[0]
    I = feature2.shape[0]

    print(range(H))
    print(range(I))

    distances = np.zeros((H, I))
    for i in range(H):
        for j in range(I):
            distances[i][j] = np.linalg.norm(feature1[i] - feature2[j])

    # Set variables for EMD calculations
    variablesList = []
    for i in range(H):
        tempList = []
        for j in range(I):
            tempList.append(LpVariable("x"+str(i)+" "+str(j), lowBound = 0))

        variablesList.append(tempList)

    problem = LpProblem("EMD", LpMinimize)

    # objective function
    constraint = []
    objectiveFunction = []
    for i in  range(H):
        for j in range(I):
            objectiveFunction.append(variablesList[i][j] * distances[i][j])

            constraint.append(variablesList[i][j])

    problem += lpSum(objectiveFunction)


    tempMin = min(sum(w1), sum(w2))
    problem += lpSum(constraint) == tempMin

    # constraints
    for i in range(H):
        constraint1 = [variablesList[i][j] for j in range(I)]
        problem += lpSum(constraint1) <= w1[i]

    for j in range(I):
        constraint2 = [variablesList[i][j] for i in range(H)]
        problem += lpSum(constraint2) <= w2[j]

    # solve
    problem.writeLP("EMD.lp")
    problem.solve(PULP_CBC_CMD(msg=False))

    flow = problem.objective.value()


    return flow / tempMin


if __name__ == "__main__":
	if(len(sys.argv) != 6):
		sys.exit("Please try running Search again. Number of arguments was different than expected.\n");
	inputFileOGS = sys.argv[1]
	queryFileOGS = sys.argv[2]
	inputLabels = sys.argv[3]
	queryLabels = sys.argv[4]
	outputFile = sys.argv[5]

	images = []
	with open(inputFileOGS, "rb") as f:
		#read metadata from file
		magicNum = int.from_bytes(f.read(4), byteorder = "big")
		numOfImages = int.from_bytes(f.read(4), byteorder = "big")
		dx = int.from_bytes(f.read(4), byteorder = "big")
		dy = int.from_bytes(f.read(4), byteorder = "big")
		
		#read images from file
		dimensions = dx*dy
		buf = f.read(dimensions * numOfImages)
		images = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
		images = images.reshape(numOfImages, dx, dy)
		f.close()

	queryImages = []
	with open(queryFileOGS, "rb") as f:
		#read metadata from file
		magicNum = int.from_bytes(f.read(4), byteorder = "big")
		numOfImages = int.from_bytes(f.read(4), byteorder = "big")
		dx = int.from_bytes(f.read(4), byteorder = "big")
		dy = int.from_bytes(f.read(4), byteorder = "big")
		
		#read images from file
		dimensions = dx*dy
		buf = f.read(dimensions * numOfImages)
		queryImages = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
		queryImages = queryImages.reshape(numOfImages, dx, dy)
		f.close()


	listofClusterSumsImage = []
	listofClusterSumsQuery = []
	listofImageCentroids = []
	listofCentroidCoordinates = []

	clusterDim = 4
	indexingSize = dx/clusterDim
	numOfClusters = len(blockshaped(queryImages[0], clusterDim, clusterDim))

	testClusterSetQuery = []
	testClusterSetImage = []
	testClusterSetQuerySum = []
	testClusterSetImageSum = []

	for index in range(0, numOfClusters):		#for every cluster
		row = int(index / indexingSize)			#cluster coordinates
		column = index % indexingSize

		clusterQ = blockshaped(queryImages[0], clusterDim, clusterDim)[index] 		#list of lists with pixel values inside a cluster
		clusterI = blockshaped(images[0], clusterDim, clusterDim)[index] 		#list of lists with pixel values inside a cluster
		testClusterSetQuery.append(clusterQ.tolist())
		testClusterSetImage.append(clusterI.tolist())
		clusterSum = 0
		for i in range (0,clusterDim):
			clusterSum += sum(clusterQ[i])			#add up all values inside a cluster (-> for every cluster)

		testClusterSetQuerySum.append(clusterSum)
		# constraint about equality between images

		clusterSum = 0
		for i in range (0,clusterDim):
			clusterSum += sum(clusterI[i])			#add up all values inside a cluster (-> for every cluster)

		# print("----------------------")

		testClusterSetImageSum.append(clusterSum)


	testClusterSetQuerySum = np.array(testClusterSetQuerySum).astype('float32') / (sum(testClusterSetQuerySum))
	testClusterSetImageSum = np.array(testClusterSetImageSum).astype('float32') / (sum(testClusterSetImageSum))
	
	difference = sum(testClusterSetQuerySum) - sum(testClusterSetImageSum)

	if difference > 0:
		clusterChosen = [idx for idx, element in enumerate(testClusterSetQuerySum) if condition(element, difference)][0]
		# print(clusterChosen)
		# print(testClusterSetQuery[clusterChosen])
		flag = 0
		for i in testClusterSetQuery[clusterChosen]:
			for j in range(len(i)):
				# print(testClusterSetQuery[clusterChosen][j])
				# print(str(testClusterSetQuery[clusterChosen][j]) + " " + str(difference))
				for index in range(len(testClusterSetQuery[clusterChosen][j])):
					if testClusterSetQuery[clusterChosen][j][index] > difference:
						testClusterSetQuery[clusterChosen][j][index] -= difference
						# print("case 1")
						flag = 1
						break
				if flag == 1:
					break
			if flag == 1:
				break

		
	elif difference < 0:
		clusterChosen = [idx for idx, element in enumerate(testClusterSetImageSum) if condition(element, difference)][0]
		# print(testClusterSetImage[clusterChosen])
		flag = 0
		for i in testClusterSetImage[clusterChosen]:
			for j in range(len(i)):
				# print(testClusterSetImage[clusterChosen][j])
				# print(str(testClusterSetImage[clusterChosen][j]) + " " + str(difference))
				for index in range(len(testClusterSetImage[clusterChosen][j])):
					if testClusterSetImage[clusterChosen][j][index] > difference:
						testClusterSetImage[clusterChosen][j][index] -= difference
						# print("case 2")
						flag = 1
						break
				if flag == 1:
					break
			if flag == 1:
				break
	# print(testClusterSetQuery[clusterChosen])
	# print(testClusterSetImage[clusterChosen])
	# print(clusterChosen)

	clusterQ = testClusterSetQuery[clusterChosen]
	clusterI = testClusterSetImage[clusterChosen]
	# sys.exit(0) 


	for index in range(0, numOfClusters):		#for every cluster

		listofCentroidCoordinates.append([(clusterDim*column) + (len(clusterQ)/2), (clusterDim*row) + (len(clusterQ)/2)])
		listofImageCentroids.append(clusterQ[int(len(clusterQ)/2)][int(len(clusterQ)/2)])		#pixel value of centroid


		# maybe break loop into 2 right here and check here for equality

		clusterSum = 0
		for i in range (0,clusterDim):
			clusterSum += sum(clusterQ[i])			#add up all values inside a cluster (-> for every cluster)

		listofClusterSumsQuery.append(clusterSum)
		# constraint about equality between images

		clusterSum = 0
		for i in range (0,clusterDim):
			clusterSum += sum(clusterI[i])			#add up all values inside a cluster (-> for every cluster)

		# print("----------------------")

		listofClusterSumsImage.append(clusterSum)


	listofClusterSumsQuery = np.array(listofClusterSumsQuery).astype('float32') / (sum(listofClusterSumsQuery))
	listofClusterSumsImage = np.array(listofClusterSumsImage).astype('float32') / (sum(listofClusterSumsImage))


	difference = sum(listofClusterSumsQuery) - sum(listofClusterSumsImage)
	print(listofClusterSumsQuery)
	print(listofClusterSumsImage)

	if difference > 0:
		clusterChosen = [idx for idx, element in enumerate(listofClusterSumsQuery) if condition(element, difference)][0]


	listofImageClusters = []

	for i in range(int(indexingSize)):		#in current row
		rowList = []
		for j in range(int(indexingSize)):
			rowList.append(listofClusterSumsQuery[i*int(indexingSize) + j])
		listofImageClusters.append(rowList)

	listofDistances = []


	for index in range(0, numOfClusters):
		currentDistances = []
		for neighbor in range(0, numOfClusters):
			# current: x = listOfImageCentroids[index][0], y = listOfImageCentroids[index][1]
			# neighbor: x = listOfImageCentroids[neighbor][0], y = listOfImageCentroids[neighbor][1]
			distance = np.linalg.norm(np.array(listofImageCentroids[index]) - np.array(listofImageCentroids[neighbor])) 
			currentDistances.append(distance)
		# distances between centroids of clusters
		listofDistances.append(currentDistances)
	# print(listofDistances)