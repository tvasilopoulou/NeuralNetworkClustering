#include <iostream>
#include <fstream>
#include <string.h>
#include <algorithm> 
#include <list> 
#include <bits/stdc++.h> 
#include "classes.hpp"
#include "clustering_functions.hpp"

/*./cluster -d train-images.idx3-ubyte -i outputDataset.txt -n clustersFile.txt -c cluster.conf -o output.txt  */

using namespace std;

int main(int argc, char * argv[]){
	int i;
	// argument check
	if (argc != 11){
		cout << ("Please try running ./cluster again. Number of arguments was different than expected.\n");
		return -1;
	}

	char * inputFileNew, * inputFileOG, * configFile, * outputFile, * clustersFile;
	int complete = 0, flag = 0;

	for (i = 1; i < argc; i++){
		if (strcmp(argv[i], "-i") == 0){
			inputFileNew=(char *)malloc(sizeof(argv[i+1] + 1)); 
			strcpy(inputFileNew, argv[i+1]);
		}

		if (strcmp(argv[i], "-d") == 0){
			inputFileOG=(char *)malloc(sizeof(argv[i+1] + 1)); 
			strcpy(inputFileOG, argv[i+1]);
		}

		else if (strcmp(argv[i], "-c") == 0){
			configFile = (char *)malloc(sizeof(argv[i+1] + 1)); 
			strcpy(configFile, argv[i+1]);
		}

		else if (strcmp(argv[i], "-o") == 0){
			outputFile = (char *)malloc(sizeof(argv[i+1] + 1)); 
			strcpy(outputFile, argv[i+1]);
		}

		else if (strcmp(argv[i], "-n") == 0){
			clustersFile = (char *)malloc(sizeof(argv[i+1] + 1)); 
			strcpy(clustersFile, argv[i+1]);
		}
		
		
	}

    /////////////////////////////////////////////////////////////////////////////
	
	// read cluster configuration file and get the values included
	int K, L, k, M, kHypercube, probes, lineConuter = 0;
	
	ifstream configF(configFile, ios::in);

	char lineCopy[50];
	string line;
	char * token;
	
	if (!configF.is_open()){
		perror("Error while opening config file");
		exit(EXIT_FAILURE);
	}

	while (getline(configF, line)){
    	strcpy(lineCopy, line.c_str());
    	token = strtok(lineCopy, ":");
    	token = strtok(NULL, ": ");
    	lineConuter++;
    	switch (lineConuter){
			case 1:
				K = atoi(token);
				break;
			case 2:
				L = atoi(token);
				break;
			case 3: 
				k = atoi(token);
				break;
			case 4:
				M = atoi(token);
				break;
			case 5:
				kHypercube = atoi(token);
				break;
			case 6:
				probes = atoi(token);
				break;
			default:
				break;
		}

    }
    configF.close();
    /////////////////////////////////////////////////////////////////////////////
	
	// read the first 4 line of input file to retireve metadata
	int w = 4000;
	int magicNum, numOfImages, dx, dy;
	ifstream inputF(inputFileOG, ios::in | ios::binary);
	for (i = 0; i < 4; i++){
		uint8_t buffer[4] ={0};
		inputF.read((char*)buffer, sizeof(buffer));
		unsigned int result = buffer[0];
		result = (result << 8) | buffer[1];
		result = (result << 8) | buffer[2];
		result = (result << 8) | buffer[3];
		switch (i){
			case 0:
				magicNum = result;
				break;
			case 1:
				numOfImages = result;
				break;
			case 2: 
				dx = result;
				break;
			case 3:
				dy = result;
				break;
			default:
				break;
		} 
	}
    /////////////////////////////////////////////////////////////////////////////

    int dimensions = dx*dy;	
	vector <Image*> imagesVector;
	uint8_t * imagesArray[numOfImages];
	uint8_t * buffer;
	i = 0;
	
	ofstream outputF(outputFile, ios::out);

	// read input file to retrieve images
	while (i!=numOfImages){
		buffer = new uint8_t[dimensions];					
		inputF.read((char*)buffer, dimensions);	
		imagesVector.push_back(new Image(i, buffer, dimensions));
		i++;

	}
	delete buffer;
	inputF.close();
    /////////////////////////////////////////////////////////////////////////////

	// run kmeans++ initialization and lloyd's for original images
	vector <Image*> initalCentroids = kMeansInitialization(imagesVector, imagesVector, L, K, numOfImages, dimensions);
	vector<Cluster *> clusterSet;
	// lloyd's algorithm for original set
	clusterSet = LloydsAlgorithm(imagesVector, imagesVector, initalCentroids, numOfImages);
	// determine objective function value for original set
    double objFuncValueS1 = objectiveFunction(clusterSet, dimensions);

	// vector<double> si = Silhouette(clusterSet, imagesVector);
	// PrintResults(outputF, clusterSet, si, imagesVector, complete);
	
    /////////////////////////////////////////////////////////////////////////////

	int numOfImagesNew = 0;
	ifstream inputFNew(inputFileNew, ios::in | ios::binary);
	for (i = 0; i < 4; i++){
		uint8_t buffer[4] ={0};
		inputFNew.read((char*)buffer, sizeof(buffer));
		unsigned int result = buffer[0];
		result = (result << 8) | buffer[1];
		result = (result << 8) | buffer[2];
		result = (result << 8) | buffer[3];
		switch (i){
			case 0:
				// magicNum = result;
				break;
			case 1:
				numOfImagesNew = result;
				break;
			case 2: 
				dx = result;
				break;
			case 3:
				dy = result;
				break;
			default:
				break;
		} 
	}
    /////////////////////////////////////////////////////////////////////////////

	int dimensionsNew = dx*dy;	
	vector <Image*> imagesVectorNew;
	uint8_t * imagesArrayNew[numOfImages];
	uint8_t * bufferNew;
	i = 0;
	
	// ofstream outputF(outputFile, ios::out);

	// read input file to retrieve images
	while (i!=numOfImages){
		bufferNew = new uint8_t[dimensionsNew];					
		inputFNew.read((char*)bufferNew, dimensionsNew);	
		imagesVectorNew.push_back(new Image(i, bufferNew, dimensionsNew));
		i++;

	}
	// delete buffer;
	// inputFNew.close();
    /////////////////////////////////////////////////////////////////////////////

	// run kmeans++ initialization and lloyd's for original images

	vector <Image*> initalCentroidsNew = kMeansInitialization(imagesVectorNew, imagesVector, L, K, numOfImages, dimensions);
	vector<Cluster *> clusterSetNew;
	// lloyd's algorithm for new vector space
	clusterSetNew = LloydsAlgorithm(imagesVectorNew, imagesVector, initalCentroidsNew, numOfImages);
	// silhouette for new vector space
	vector<double> siNew = Silhouette(clusterSetNew, imagesVectorNew);

	// append to output file
	outputF << "NEW SPACE" << endl;
    double objFuncValueS2 = objectiveFunction(clusterSetNew, dimensions);
	PrintResults(outputF, clusterSetNew, siNew, imagesVectorNew, complete);
	outputF << "Value of Objective Function: " << objFuncValueS2 << endl;

	outputF << "ORIGINAL SPACE" << endl;
	vector<double> si = Silhouette(clusterSet, imagesVector);
	PrintResults(outputF, clusterSet, si, imagesVector, complete);
	outputF << "Value of Objective Function: " << objFuncValueS1 << endl;

    /////////////////////////////////////////////////////////////////////////////

	ifstream clusterF(clustersFile, ios::in);

	// read cluster file
	char charBuffer[60000];
	string strLine;
	// char * token;
	cout << clustersFile << endl;
	if (!clusterF.is_open()){
		perror("Error while opening cluster file");
		exit(EXIT_FAILURE);
	}

	vector<Cluster *> clusters;
	int index = 0; 

	for (int c = 0; c < 10; c++){
		Cluster * newCluster = new Cluster(NULL);
		clusters.push_back(newCluster);
	}

	// imagesVector.resize(1000);
	// for classes as clusters, call half lloyd's algorithm
    clusters = HalfLloydsAlgorithm(imagesVector, numOfImages, clusters);

	while (getline(clusterF, strLine)){
    	strcpy(charBuffer, strLine.c_str());
    	token = strtok(charBuffer, "-");
    	token = strtok(NULL, "{");
    	index = atoi(token);
    	token = strtok(NULL, "}");
		token = strtok(token, ",");
    	lineConuter++;
    	while(token!=NULL){
    		// initialize clusters and image distances distances from centroids
	    	(clusters[index]->getImagesVector())->push_back(imagesVector[atoi(token)]->getVal());
	    	double mDist = manhattan(imagesVector[atoi(token)]->getVal(), clusters[index]->getCluster()->getVal(), dimensions);
	    	imagesVector[atoi(token)]->setMinDist(mDist);
	    	imagesVector[atoi(token)]->setCluster(index);
    		token = strtok(NULL, " , ");

	    }
    }


    // find objectiveFunction
    double objFuncValue = objectiveFunction(clusters, dimensions);


	si = Silhouette(clusters, imagesVector);
	outputF << "CLASSES AS CLUSTERS " << endl;
	outputF << "Silhouette: [" << endl;
	double sum = 0.0;
	for(int i=0; i<si.size(); i++){
		sum += si[i];
		outputF << si[i] << ", ";
	}
	outputF << sum << "]" << endl << "Value of Objective Function: " << objFuncValue << endl;


    clusterF.close();
    
    /////////////////////////////////////////////////////////////////////////////


	// clear vectors, free variables and return
	outputF.close();
	free(inputFileNew);
	free(inputFileOG);
	free(clustersFile);
	free(outputFile);
	free(configFile);
	
	return 0;
}


