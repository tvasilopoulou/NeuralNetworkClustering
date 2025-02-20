#include <iostream>
#include <bits/stdc++.h> 
#include <cmath>
#include <vector>
#include "header.hpp"
using namespace std; 

// function to calculate manhattan distance
int manhattanDistance(uint8_t * tempImage, uint8_t * qImage, int size){
	int distance = 0;
	for (int i=0; i<size; i++)
	{
		distance += abs(qImage[i] - tempImage[i]);
	}
	return distance;
}

bool operator<(Values & x, Values & y){
    return x.getIndex() < y.getIndex();
}


/*brute force calculations*/
int CalculateDistances(uint8_t * qImage, int dimensions, vector <uint8_t *> imageVector, int * pos){
	vector <int> distances;
	for(auto &image : imageVector)
	{
		if(qImage != image) 
		{
			distances.push_back(manhattanDistance(qImage, image, dimensions));
		}
	}

	// find the index of the minimum
	int min = distances[0];
	for (int i = 0; i < distances.size(); i++)
	{
		if (distances[i] < min)
		{
			min = distances[i];
			*pos = i;
		}
	}

	// sort(distances.begin(),distances.end());
	return min;
}

/*geeks4geeks*/
int hammingDistance(int n1, int n2) { 
	int x = n1 ^ n2; 
	int setBits = 0; 
	while (x > 0) { 
		setBits += x & 1; 
		x >>= 1; 
	} 
	return setBits; 
} 

// function to check if an image exists in a vector of images
int exists(Values * neighbors, int id , int size){
	for(int i = 0; i < size; i++){
		if(neighbors[i].getHashResult() == id) return i;
	}
	return -1;
}

