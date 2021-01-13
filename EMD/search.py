import numpy as np
from pulp import *
import time
import sys

# https://stackoverflow.com/questions/16856788/slice-2d-array-into-smaller-2d-arrays
def blockshaped(arr, nrows, ncols):
	"""
	Return an array of shape (n, nrows, ncols) where
	n * nrows * ncols = arr.size

	If arr is a 2D array, the returned array should look like n subblocks with
	each subblock preserving the "physical" layout of arr.
	"""
	h, w = arr.shape
	assert h % nrows == 0, "{} rows is not evenly divisible by {}".format(h, nrows)
	assert w % ncols == 0, "{} cols is not evenly divisible by {}".format(w, ncols)
	return (arr.reshape(h//nrows, nrows, -1, ncols).swapaxes(1,2).reshape(-1, nrows, ncols))

# append to best neighbors if better than already existing ones
def addToNeighbors(listofNeighbors, newNeighbor, imgIndex):
	# if distance is less than what is currently available, append to the end and sort
	if newNeighbor < listofNeighbors[-1][1]:
		listofNeighbors[-1][1] = newNeighbor
		listofNeighbors[-1][0] = imgIndex
		listofNeighbors.sort(key = lambda x: x[1])
	return listofNeighbors


def manhattanNeighbors(queryImages, images, querySetSize, imageSetSize): 
	# exhaustively calculate best neighbors by manhattan distance
	neighborsPerQueryImage = []
	start = time.time()
	for i in range(querySetSize):
		nearestNeighbors = [[-1,100000.0], [-1,100000.0], [-1,100000.0], [-1,100000.0], [-1,100000.0], [-1,100000.0], [-1,100000.0], [-1,100000.0], [-1,100000.0], [-1,100000.0]]
		flatQuery = queryImages[i].flatten()
		for j in range(imageSetSize):
			distance = 0.0
			flatImage = images[j].flatten()
			for index in range(len(flatQuery)):
				distance += abs(flatQuery[index] - flatImage[index])
			nearestNeighbors = addToNeighbors(nearestNeighbors, distance, j)
		neighborsPerQueryImage.append(nearestNeighbors)
	end = time.time()
	timeElapsed = end - start 
	return timeElapsed, neighborsPerQueryImage


def calculateCorrectness(neighborsPerQueryImage, queryImages, images, trainLabels, testLabels, querySetSize, imageSetSize):
	# if neighbor is same as query image => count as correct 
	correctLabels = 0
	for i in range(querySetSize):
		queryLabel = testLabels[i]

		for j in range(len(neighborsPerQueryImage[i])):
			neighborIndex = neighborsPerQueryImage[i][j][0]
			if trainLabels[neighborIndex] == queryLabel:
				correctLabels += 1
	# correct predictions/ set size 
	percentage = correctLabels / querySetSize
	return percentage

def EMD(queryImages, images, querySetSize, imageSetSize):
	listofClusterSumsImage = []
	listofClusterSumsQuery = []
	listofImageCentroids = []
	listofCentroidCoordinates = []

	# for 49 (7x7) clusters of size 4x4
	clusterDim = 4
	indexingSize = dx/clusterDim
	numOfClusters = len(blockshaped(queryImages[0], clusterDim, clusterDim))

	testClusterSetQuery = []
	testClusterSetImage = []
	testClusterSetQuerySum = []
	testClusterSetImageSum = []

	start = time.time()

	neighborsPerQueryImage = []
	neighborsPerQueryImageIndexes = []

	for currIndex in range(querySetSize):
		# for every query image, find 10 nearest neighbors as requested
		nearestNeighbors = [[-1,100000.0], [-1,100000.0], [-1,100000.0], [-1,100000.0], [-1,100000.0], [-1,100000.0], [-1,100000.0], [-1,100000.0], [-1,100000.0], [-1,100000.0]]
		for imgIndex in range(imageSetSize):
			# for every training image
			for index in range(0, numOfClusters):
				#for every cluster calculate coordinates
				row = int(index / indexingSize)
				column = index % indexingSize

				# curent cluster representation as list of lists
				clusterQ = blockshaped(queryImages[currIndex], clusterDim, clusterDim)[index] 
				clusterI = blockshaped(images[imgIndex], clusterDim, clusterDim)[index] 

				# contain full images made by clusters
				testClusterSetQuery.append(clusterQ.tolist())
				testClusterSetImage.append(clusterI.tolist())

				#add up all values inside a cluster for every cluster
				clusterSum = 0
				for i in range (0,clusterDim):
					clusterSum += sum(clusterQ[i])
				testClusterSetQuerySum.append(clusterSum)


				#add up all values inside a cluster for every cluster
				clusterSum = 0
				for i in range (0,clusterDim):
					clusterSum += sum(clusterI[i])
				testClusterSetImageSum.append(clusterSum)



			# normalize image full sum to 1, for both query and train images
			testClusterSetQuerySum = np.array(testClusterSetQuerySum).astype('float32') / (sum(testClusterSetQuerySum))
			testClusterSetImageSum = np.array(testClusterSetImageSum).astype('float32') / (sum(testClusterSetImageSum))
			
			# find difference between the two images' full brightness
			difference = sum(testClusterSetQuerySum) - sum(testClusterSetImageSum)

			# if query image brighter, find first avalable cluster to decrement from and do so
			if difference > 0:
				# find non zero cluster
				clusterChosen = [idx for idx, element in enumerate(testClusterSetQuerySum) if element > difference][0]
				row = int(clusterChosen / indexingSize)
				column = int(clusterChosen % indexingSize)
				flag = 0
				for i in testClusterSetQuery[clusterChosen]:
					for j in range(len(i)):
						for index in range(len(testClusterSetQuery[clusterChosen][j])):
							# find non zero pixel to reduce 
							if testClusterSetQuery[clusterChosen][j][index] > difference:
								for dim in range(0, clusterDim):
									# find pixel in original image to overwrite
									if queryImages[currIndex][row + dim][column] == testClusterSetQuery[clusterChosen][j][index]:
										queryImages[currIndex][row + dim][column] -= difference * sum(testClusterSetQuerySum)

									elif queryImages[currIndex][row][column + dim] == testClusterSetQuery[clusterChosen][j][index]:
										queryImages[currIndex][row][column + dim] -= difference * sum(testClusterSetQuerySum)

									elif queryImages[currIndex][row + dim][column + dim] == testClusterSetQuery[clusterChosen][j][index]:
										queryImages[currIndex][row + dim][column + dim] -= difference * sum(testClusterSetQuerySum)
									break
								testClusterSetQuery[clusterChosen][j][index] -= difference
								flag = 1
								break
						if flag == 1:
							break
					if flag == 1:
						break

			# else, do the same but for training image
			elif difference < 0:
				clusterChosen = [idx for idx, element in enumerate(testClusterSetImageSum) if element > (-1) * difference][0]
				row = int(clusterChosen / indexingSize)
				column = int(clusterChosen % indexingSize)
				flag = 0
				for i in testClusterSetImage[clusterChosen]:
					for j in range(len(i)):
						for index in range(len(testClusterSetImage[clusterChosen][j])):
							if testClusterSetImage[clusterChosen][j][index] > (-1) * difference:
								for dim in range(0, clusterDim):
									# find pixel in original image to overwrite
									if queryImages[currIndex][row + dim][column] == testClusterSetQuery[clusterChosen][j][index]:
										queryImages[currIndex][row + dim][column] += difference
									elif queryImages[currIndex][row][column + dim] == testClusterSetQuery[clusterChosen][j][index]:
										queryImages[currIndex][row][column + dim] += difference
									elif queryImages[currIndex][row + dim][column + dim] == testClusterSetQuery[clusterChosen][j][index]:
										queryImages[currIndex][row + dim][column + dim] += difference
								testClusterSetQuery[clusterChosen][j][index] += difference
								flag = 1
								break
						if flag == 1:
							break
					if flag == 1:
						break

			# continue with set clusters, the ones of certain index

			for index in range(0, numOfClusters):
				row = int(index / indexingSize)
				column = index % indexingSize

				# repeat process for new image brightness
				clusterQ = blockshaped(queryImages[currIndex], clusterDim, clusterDim)[index]
				clusterI = blockshaped(images[imgIndex], clusterDim, clusterDim)[index]
				listofCentroidCoordinates.append([(clusterDim*column) + (len(clusterQ)/2), (clusterDim*row) + (len(clusterQ)/2)])
				listofImageCentroids.append(clusterQ[int(len(clusterQ)/2)][int(len(clusterQ)/2)])

				clusterSum = 0
				for i in range (0,clusterDim):
					clusterSum += sum(clusterQ[i])

				listofClusterSumsQuery.append(clusterSum)
				# constraint about equality between images

				clusterSum = 0
				for i in range (0,clusterDim):
					clusterSum += sum(clusterI[i])			#add up all values inside a cluster (-> for every cluster)

				listofClusterSumsImage.append(clusterSum)

			# images' brightness comparison should now yield difference = 0
			listofClusterSumsQuery = np.array(listofClusterSumsQuery).astype('float32') / (sum(listofClusterSumsQuery))
			listofClusterSumsImage = np.array(listofClusterSumsImage).astype('float32') / (sum(listofClusterSumsImage))

			listofDistances = []
			# find distances beween clusters
			for index in range(0, numOfClusters):
				currentDistances = []
				for neighbor in range(0, numOfClusters):
					# current: x = listOfImageCentroids[index][0], y = listOfImageCentroids[index][1]
					# neighbor: x = listOfImageCentroids[neighbor][0], y = listOfImageCentroids[neighbor][1]
					distance = np.linalg.norm(np.array(listofCentroidCoordinates[index]) - np.array(listofCentroidCoordinates[neighbor])) 
					currentDistances.append(distance)
				# distances between centroids of clusters
				listofDistances.append(currentDistances)


			# create the lp flows, set them as >=0
			listofFlows = []
			for i in range(len(listofImageCentroids)):
				clusterFlows = []
				for j in range(len(listofImageCentroids)):
					clusterFlows.append(LpVariable("F"+str(i)+"_"+str(j), lowBound = 0))

				listofFlows.append(clusterFlows)

			# initialize lp problem
			problem = LpProblem("EMD", LpMinimize)

			constraint = []
			objectiveFunction = []

			for i in range(numOfClusters):
				for j in range(numOfClusters):
					objectiveFunction.append(listofFlows[i][j] * listofDistances[i][j])
					# flatten flows
					constraint.append(listofFlows[i][j])

			problem += lpSum(objectiveFunction)

			# make sure the constraint is based on the lowest image brightness in case 
			tempMin = min(sum(listofClusterSumsQuery), sum(listofClusterSumsImage))
			problem += lpSum(constraint) == tempMin


			# lp problem constraints as set by the slides
			for i in range(numOfClusters):
				constraint1 = [listofFlows[i][j] for j in range(numOfClusters)]
				problem += lpSum(constraint1) <= listofClusterSumsQuery[i]

			for j in range(numOfClusters):
				constraint2 = [listofFlows[i][j] for i in range(numOfClusters)]
				problem += lpSum(constraint2) <= listofClusterSumsImage[j]

			# solve without messages print
			problem.solve(PULP_CBC_CMD(msg=False))

			listofClusterSumsImage = []
			listofClusterSumsQuery = []
			listofImageCentroids = []
			listofCentroidCoordinates = []

			testClusterSetQuery = []
			testClusterSetImage = []
			testClusterSetQuerySum = []
			testClusterSetImageSum = []

			# problem solution
			flow = problem.objective.value()
			# if better than already existent, append
			nearestNeighbors = addToNeighbors(nearestNeighbors, flow, imgIndex)

		neighborsPerQueryImage.append(nearestNeighbors)
	end = time.time()
	# return timeElapsed and best neighbors for each image
	return (end - start)/querySetSize, neighborsPerQueryImage


if __name__ == "__main__":
	if(len(sys.argv) != 6):
		sys.exit("Please try running Search again. Number of arguments was different than expected.\n");
	inputFileOGS = sys.argv[1]
	queryFileOGS = sys.argv[2]
	inputLabels = sys.argv[3]
	queryLabels = sys.argv[4]
	outputFile = sys.argv[5]


	# read training images into buffer
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

	# read query images into buffer
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


	trainLabels = []
	with open(inputLabels, "rb") as f:
		#read metadata from file
		magicNum = int.from_bytes(f.read(4), byteorder = "big")
		numOfImages = int.from_bytes(f.read(4), byteorder = "big")
		
		#read labels from file
		buf = f.read(numOfImages)
		trainLabels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
		f.close()
	trainLabels = np.array(trainLabels)

	testLabels = []
	with open(queryLabels, "rb") as f:
		#read metadata from file
		magicNum = int.from_bytes(f.read(4), byteorder = "big")
		numOfImages = int.from_bytes(f.read(4), byteorder = "big")

		#read labels from file
		buf = f.read(numOfImages)
		testLabels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
		f.close()

	testLabels = np.array(testLabels)

	# testing chunk for datasets
	querySetSize = 10
	imageSetSize = 100

	# calculate nearest neighbors by EMD
	neighborsPerQueryImage = []
	timeElapsed, neighborsPerQueryImage = EMD(queryImages, images, querySetSize, imageSetSize)
	emdCorrectness = calculateCorrectness(neighborsPerQueryImage, queryImages, images, trainLabels, testLabels, querySetSize, imageSetSize)

	# write results to output file
	f = open(outputFile, "a")
	f.write("Average Correct Search Results EMD: " + str(emdCorrectness) + "\n")
	f.write("Average Time Elapsed in EMD Neighbor Search : " + str(timeElapsed) + "\n")
	f.close()

	# calculate nearest neighbors by manhattan metric
	neighborsPerQueryImage = []
	timeElapsed, neighborsPerQueryImage = manhattanNeighbors(queryImages, images, querySetSize, imageSetSize)
	trueCorrectness = calculateCorrectness(neighborsPerQueryImage, queryImages, images, trainLabels, testLabels, querySetSize, imageSetSize)

	# write results to output file
	f = open(outputFile, "a")
	f.write("Average Correct Search Results MANHATTAN: " + str(trueCorrectness) + "\n")
	f.write("Average Time Elapsed in MANHATTAN Neighbor Search : " + str(timeElapsed) + "\n")
	f.close()
