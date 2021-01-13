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
	printf("Command line arguments checked.\n");

	/*END OF ARGUMENT CHECK*/
	/*----------------------------------------------------------------------------------------------------------------------------------*/

	// read the first 4 bytes of original space input file to retrieve metadata
	int w = 400;
	int magicNum, numOfImages, dx, dy;
	printf("Reading original space input file to retrieve metadata:\n");
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
	printf("magicNum = %d \nnumOfImages = %d\ndx = %d\ndy = %d\n", magicNum, numOfImages, dx, dy);
	//end of metadata retireval
	/*----------------------------------------------------------------------------------------------------------------------------------*/

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

	// read the first 4 bytes of new space input file to retrieve metadata
	int magicNum1, numOfImages1, dx1, dy1;

	ifstream newInputF(newInputFile, ios::in | ios::binary);
	printf("Reading new space input file to retrieve metadata:\n");

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
	printf("magicNum1 = %d \nnumOfImages1 = %d\ndx1 = %d\ndy1 = %d\n", magicNum1, numOfImages1, dx1, dy1);
	//end of metadata retireval
	/*----------------------------------------------------------------------------------------------------------------------------------*/

	int dimensions1 = dx1*dy1;
	
	vector <uint8_t *> newImagesVector;
	uint8_t * newBuffer;
	
	// read the new space input file to get images
	for(i = 0; i < numOfImages1; i++){
		newBuffer = new uint8_t[dimensions1];
		newInputF.read((char *)newBuffer, dimensions1);
		newImagesVector.push_back(newBuffer);
	}

	newInputF.close();
	/*----------------------------------------------------------------------------------------------------------------------------------*/
	
	printf("Reading original space query file to retrieve metadata:\n");
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
	printf("magicNum = %d \nnumOfImages = %d\ndx = %d\ndy = %d\n", magicNum, numOfImages, dx, dy);
	//end of metadata retireval
	/*----------------------------------------------------------------------------------------------------------------------------------*/

	// read the first 4 bytes of new query file to retrieve metadata
	printf("Reading new space query file to retrieve metadata:\n");
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
	printf("magicNum = %d \nnumOfImages = %d\ndx = %d\ndy = %d\n", magicNum1, numOfImages1, dx1, dy1);
	//end of metadata retireval
	/*----------------------------------------------------------------------------------------------------------------------------------*/
	
	ofstream outputF;
	outputF.open(outputFile);

	// start routine to print results to output file
	printf("Starting nearest neighbor search routine.\n");

	int sumOfDistancesLSH = 0;
	int sumOfDistancesNN = 0;
	int sumOfDistancesTrue = 0;
	
	double durationReduced = 0.0;
	double durationLSH = 0.0;
	double durationTrue = 0.0;

	for(i = 0; i < numOfImages; i++)
	{
		// compute time needed to find nearest neighbor using brute force
		auto t1True = std::chrono::high_resolution_clock::now();
		buffer = new uint8_t[dimensions];
		queryF.read((char*)buffer, dimensions);
		int pos = 0;
		int realDists = CalculateDistances(&buffer[i], dimensions, imagesVector, &pos);
		auto t2True = std::chrono::high_resolution_clock::now();
		durationTrue += std::chrono::duration_cast<std::chrono::milliseconds>(t2True - t1True).count();

		// compute time needed to find nearest neighbor using lsh
		auto t1LSH = std::chrono::high_resolution_clock::now();
		Values * neighbors = hashMap->ANN(buffer);
		auto t2LSH = std::chrono::high_resolution_clock::now();
		durationLSH += std::chrono::duration_cast<std::chrono::milliseconds>(t2LSH - t1LSH).count();
		
		// compute time needed to find nearest neighbor in the reduced space
		auto t1Reduced = std::chrono::high_resolution_clock::now();
		newBuffer = new uint8_t[dimensions1];
		newQueryF.read((char*)newBuffer, dimensions1);
		int pos1 = 0;
		int reducedDists = CalculateDistances(&newBuffer[i], dimensions1, newImagesVector, &pos1);
		int reduced = manhattanDistance(imagesVector[pos1], &buffer[i], dimensions); 
		auto t2Reduced = std::chrono::high_resolution_clock::now();
		durationReduced += std::chrono::duration_cast<std::chrono::milliseconds>(t2Reduced - t1Reduced).count();
		
		outputF << endl << "Query: " << i << endl;
		
		outputF << "Nearest neighbor Reduced: " << pos1 << endl; 
		outputF << "Nearest neighbor LSH: " << neighbors[0].getHashResult() << endl; 
		outputF << "Nearest neighbor: " << pos << endl; 
		
		outputF << "distanceReduced: " << reduced << endl;
		outputF << "distanceLSH: " << neighbors[0].getIndex() << endl; 
		outputF << "distanceTrue: " << realDists << endl;

		
		sumOfDistancesLSH += neighbors[0].getIndex();
		sumOfDistancesNN +=  reduced;
		sumOfDistancesTrue += realDists;
	
	}
	// close opened files
	queryF.close();
	newQueryF.close();
	
	outputF << endl;

	outputF << "tReduced: " << durationReduced << "ms" << endl;
	outputF << "tLSH: " << durationLSH <<  "ms" << endl;
	outputF << "tTrue: " << durationTrue <<  "ms" << endl;

	// calculate approximation factor
	double af1 = 0.0;
	double af2 = 0.0;


	double meanLSH, meanNN, meanTrue;
	meanLSH = (double)((double)sumOfDistancesLSH/(double)numOfImages);
	meanNN = (double)((double)sumOfDistancesNN/(double)numOfImages);
	meanTrue = (double)((double)sumOfDistancesTrue/(double)numOfImages);

	af1 = meanLSH/meanTrue;
	outputF << "Approximation Factor LSH: " << af1 << endl;
	
	af2 = meanNN/meanTrue;
	outputF << "Approximation Factor Reduced: " << af2 << endl;
	
	outputF.close();

	printf("Routine finished. For results check %s\n", outputFile);

	// delete/free dynamically allocated variables/objects
	delete hashMap;
	delete buffer;
	delete newBuffer;
	free(originalInputFile);
	free(originalQueryFile);
	free(newInputFile);
	free(newQueryFile);
	free(outputFile);

	return 0;
}

