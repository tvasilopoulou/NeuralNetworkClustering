#include <iostream>
#include <fstream>
#include <string.h>
#include <algorithm> 
#include <list> 
#include <vector>
#include <bits/stdc++.h> 
#include "header.hpp"
#include "funcHeader.hpp"

/*./search -d train-images.idx3-ubyte -i 1 -q t10k-images.idx3-ubyte -s 2 -k 4 -L 5 -o output.txt*/

using namespace std;

int main(int argc, char * argv[]){
	
	if (argc != 15)
	{
		cout << ("Please try running search again. Number of arguments was different than expected.\n");
		return -1;
	}
	char * originalInputFile, * newInputFile, * originalQueryFile, * newQueryFile, * outputFile;
	int k = 0, L = 0, N = 1;
	int i, j;
	
	for (i = 0; i < argc; i++)
	{
		if (strcmp(argv[i], "-d") == 0)
		{
			originalInputFile = (char *)malloc(sizeof(argv[i+1] + 1)); 
			strcpy(originalInputFile, argv[i+1]);
		}
		else if (strcmp(argv[i], "-i") == 0)
		{
			newInputFile = (char *)malloc(sizeof(argv[i+1] + 1)); 
			strcpy(newInputFile, argv[i+1]);
		}
		else if (strcmp(argv[i], "-q") == 0)
		{
			originalQueryFile = (char *)malloc(sizeof(argv[i+1] + 1)); 
			strcpy(originalQueryFile, argv[i+1]);
		}
		else if (strcmp(argv[i], "-s") == 0)
		{
			newQueryFile = (char *)malloc(sizeof(argv[i+1] + 1)); 
			strcpy(newQueryFile, argv[i+1]);
		}
		else if (strcmp(argv[i], "-k") == 0)
		{
			k = atoi(argv[i+1]);
		}
		else if (strcmp(argv[i], "-L") == 0)
		{
			L = atoi(argv[i+1]);
		}
		else if (strcmp(argv[i], "-o") == 0)
		{
			outputFile = (char *)malloc(sizeof(argv[i+1] + 1)); 
			strcpy(outputFile, argv[i+1]);
		}
	}

	printf("%s %s %s %s %d %d %s\n", originalQueryFile, newInputFile, originalQueryFile, newQueryFile, k, L, outputFile);
	// return 0;	
	/*END OF ARGUMENT CHECK*/
	/*----------------------------------------------------------------------------------------------------------------------------------*/

	// read the first 4 bytes to retrieve metadata
	int w = 400;
	int magicNum, numOfImages, dx, dy;
	
	ifstream inputF(originalInputFile, ios::in | ios::binary);
	for(i = 0; i < 4; i++)
	{
		uint8_t buffer[4] = {0};
		inputF.read((char*)buffer, sizeof(buffer));
		unsigned int result = buffer[0];
		result = (result << 8) | buffer[1];
		result = (result << 8) | buffer[2];
		result = (result << 8) | buffer[3];
		switch(i)
		{
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
	//end of metadata retireval
	/*----------------------------------------------------------------------------------------------------------------------------------*/
	printf("%d %d %d %d\n", magicNum, numOfImages, dx, dy);


	int fixedInd = numOfImages/16;
	int dimensions = dx*dy;

	// create the hash map for lsh
	HashMap * hashMap = new HashMap(L, fixedInd, k, dimensions, w, N);
	
	vector <uint8_t *> imagesVector;
	uint8_t * buffer;
	
	// read the input file to get images
	for(i = 0; i < numOfImages; i++){
		buffer = new uint8_t[dimensions];
		inputF.read((char *)buffer, dimensions);
		imagesVector.push_back(buffer);
		for(j = 0; j < L; j++){
			hashMap->getHashTableByIndex(j)->hashFunctionG(w, dimensions, buffer, i); 
		}
	}

	inputF.close();

	/*----------------------------------------------------------------------------------------------------------------------------------*/

	// read the first 4 bytes of query file to retrieve metadata
	ifstream queryF(originalQueryFile, ios::in | ios::binary);
	for(i = 0; i < 4; i++)
	{
		uint8_t buffer[4] = {0};
		queryF.read((char*)buffer, sizeof(buffer));
		unsigned int result = buffer[0];
		result = (result << 8) | buffer[1];
		result = (result << 8) | buffer[2];
		result = (result << 8) | buffer[3];
		switch(i){
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
	//end of metadata retireval
	/*----------------------------------------------------------------------------------------------------------------------------------*/
	printf("%d %d %d %d\n", magicNum, numOfImages, dx, dy);


	// read the first 4 bytes to retrieve metadata
	int magicNum1, numOfImages1, dx1, dy1;

	ifstream newInputF(newInputFile, ios::in | ios::binary);

	for(i = 0; i < 4; i++){
		uint8_t buffer[4] = {0};
		newInputF.read((char*)buffer, sizeof(buffer));
		unsigned int result = buffer[0];
		result = (result << 8) | buffer[1];
		result = (result << 8) | buffer[2];
		result = (result << 8) | buffer[3];
		switch(i){
			case 0:
				magicNum1 = result;
				break;
			case 1:
				numOfImages1 = result;
				break;
			case 2: 
				dx1 = result;
				break;
			case 3:
				dy1 = result;
				break;
			default:
				break;
		} 
	}
	//end of metadata retireval
	/*----------------------------------------------------------------------------------------------------------------------------------*/
	printf("%d %d %d %d\n", magicNum1, numOfImages1, dx1, dy1);

	int dimensions1 = dx1*dy1;
	
	vector <uint8_t *> newImagesVector;
	uint8_t * newBuffer;
	
	// read the input file to get images
	for(i = 0; i < numOfImages1; i++){
		newBuffer = new uint8_t[dimensions1];
		newInputF.read((char *)newBuffer, dimensions1);
		newImagesVector.push_back(newBuffer);
	}
	newInputF.close();

	// read the first 4 bytes of query file to retrieve metadata
	ifstream newQueryF(newQueryFile, ios::in | ios::binary);
	for(i = 0; i < 4; i++)
	{
		uint8_t buffer[4] = {0};
		newQueryF.read((char*)buffer, sizeof(buffer));
		unsigned int result = buffer[0];
		result = (result << 8) | buffer[1];
		result = (result << 8) | buffer[2];
		result = (result << 8) | buffer[3];
		switch(i){
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
	//end of metadata retireval
	/*----------------------------------------------------------------------------------------------------------------------------------*/
	printf("%d %d %d %d\n", magicNum, numOfImages, dx, dy);

	
	// start routine to print results to output file
	ofstream outputF;
	outputF.open(outputFile);

	double durationReduced = 0.0;
	double durationLSH = 0.0;
	double durationTrue = 0.0;

	vector <int> approximmateDistancesLSH; 
	vector <int> approximmateDistancesReduced; 
	vector <int> trueDistances; 

	auto t1True = std::chrono::high_resolution_clock::now();
	auto t1LSH = std::chrono::high_resolution_clock::now();
	auto t1Reduced = std::chrono::high_resolution_clock::now();

	int sumOfDistancesLSH = 0;
	int sumOfDistancesNN = 0;
	int sumOfDistancesTrue = 0;
	
	for(i = 0; i < numOfImages; i++)
	{	
		buffer = new uint8_t[dimensions];
		queryF.read((char*)buffer, dimensions);

		int pos = 0;
		vector <int> realDists = CalculateDistances(buffer, dimensions, imagesVector, &pos);
		realDists.resize(N);
		trueDistances.push_back(realDists[0]);
		auto t2True = std::chrono::high_resolution_clock::now();
		durationTrue = std::chrono::duration_cast<std::chrono::milliseconds>(t2True - t1True).count();

		Values * neighbors = hashMap->ANN(buffer);
		approximmateDistancesLSH.push_back(neighbors[0].getIndex());
		auto t2LSH = std::chrono::high_resolution_clock::now();
		durationLSH = std::chrono::duration_cast<std::chrono::milliseconds>(t2LSH - t1LSH).count();
		
		newBuffer = new uint8_t[dimensions1];
		newQueryF.read((char*)newBuffer, dimensions1);
		int pos1 = 0;
		vector <int> reducedDists = CalculateDistances(newBuffer, dimensions1, newImagesVector, &pos1);
		reducedDists.resize(N);
		approximmateDistancesReduced.push_back(reducedDists[0]);
		auto t2Reduced = std::chrono::high_resolution_clock::now();
		durationReduced = std::chrono::duration_cast<std::chrono::milliseconds>(t2Reduced - t1Reduced).count();
		
		// cout << "--" << neighbors->getIndex() << " " << neighbors->getHashResult() << endl;

		outputF << endl << "Query: " << i << endl;
		
		outputF << "Nearest neighbor Reduced: " << pos1 << endl; 
		outputF << "Nearest neighbor LSH: " << neighbors[0].getHashResult() << endl; 
		outputF << "Nearest neighbor: " << pos << endl; 
		
		outputF << "distanceReduced: " << reducedDists[0] << endl;
		outputF << "distanceLSH: " << neighbors[0].getIndex() << endl; 
		outputF << "distanceTrue: " << realDists[0] << endl;

		sumOfDistancesLSH += neighbors[0].getIndex();
		sumOfDistancesNN += reducedDists[0];
		sumOfDistancesTrue += realDists[0];
		
		outputF << "tReduced: " << durationReduced << endl;
		outputF << "tLSH: " << durationLSH << endl;
		outputF << "tTrue: " << durationTrue << endl;

	}
	queryF.close();
	newQueryF.close();


	printf("%ld %ld %ld\n", approximmateDistancesReduced.size(), approximmateDistancesLSH.size(), trueDistances.size());

	double af1 = 0.0;
	double af2 = 0.0;

	outputF << endl;


	af1 = ((double)(sumOfDistancesLSH)/(double)(sumOfDistancesTrue))/((double)(numOfImages));
	outputF << "Approximation Factor LSH: " << af1 << endl;
	
	af2 = ((double)(sumOfDistancesNN)/(double)(sumOfDistancesTrue))/((double)(numOfImages));
	outputF << "Approximation Factor Reduced: " << af2 << endl;
	
	outputF.close();

	delete hashMap;
	delete buffer;
	free(originalInputFile);
	free(originalQueryFile);
	free(newInputFile);
	free(newQueryFile);
	free(outputFile);
	return 0;
}

