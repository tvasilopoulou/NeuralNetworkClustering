#include <fstream>
#include <iostream>
#include <fstream>
#include <string.h>
#include <iterator> 
#include <map>
#include <vector>
#include <algorithm>
#include <bits/stdc++.h> 

using namespace std; 
  
// Comparator function to sort pairs 
// according to second value 
bool cmp(pair<int, int>& a, 
         pair<int, int>& b) 
{ 
    return a.second < b.second; 
} 
  
// Function to sort the map according 
// to value in a (key-value) pairs 
void sortMap(map<int, int>& M) 
{ 
  
    // Declare vector of pairs 
    vector<pair<int, int> > A; 
  
    // Copy key-value pair from Map 
    // to vector of pairs 
    for (auto& it : M) { 
        A.push_back(it); 
    } 
  
    // Sort using comparator function 
    sort(A.begin(), A.end(), [](const pair<int, int>& l, const pair<int, int>& r) {
                if (l.second != r.second)
                    return l.second < r.second;
 
                return l.first < r.first;
            });
  	
	int numberOfClusters = 0;
	
	for (auto const &pair: A) 
	{
		numberOfClusters = pair.second;    	
    }

    printf("%d clusters\n", numberOfClusters);

	vector <int> clusterSize;  
	
	for (int i = 0; i <= numberOfClusters; i++)
    {
		clusterSize.push_back(0);
   	} 

    for (int i = 0; i <= numberOfClusters; i++)
    {
    	for (auto const &pair: A) 
		{
			if(pair.second == i)   	
			{
				clusterSize[i]++;
			}
	    }
   	}

  
    ofstream outputF;
	outputF.open("clusters.txt");

   	for (int i = 0; i <= numberOfClusters; i++)
    {
    	int c = 0;
    	printf("%d %d\n",i, clusterSize[i]);
    	outputF << "Cluster-" << i+1 << " { size:" << clusterSize[i] << ", ";
    	for (auto const &pair: A) 
		{
			if(pair.second == i)
			{
				c++;
				if (c < clusterSize[i])
				{
	        		outputF << pair.first << ", ";
				}
				else
				{
	        		outputF << pair.first << "}" << endl;
				}
			}
	    }
   	}


	// for (auto const &pair: A) 
	// {
 //        std::cout << '{' << pair.second << "," << pair.first << '}' << '\n';
 //    }

} 

int main(int argc, char const *argv[])
{
	// check number of arguments 
    if(argc != 2)
    {
		printf("quic.c Error: Invalid argument format. Please try ./fileTransform fileName.\n");
		return -1;
	}
    ////////////////////////////////////////////////////////////////////

	char * labelsFile = (char *)malloc(sizeof(char)*strlen(argv[1] + 1));
	strcpy(labelsFile, argv[1]);

	printf("%s\n", labelsFile);

	
	ifstream file;
	
	map<int, int> clusterDistr;
	
	file.open(labelsFile);
	
    string line;
    int i = 1;

    while (getline(file, line)) 
    {
        clusterDistr.insert(pair<int, int>(i, atoi(line.c_str())));
        i++;
    }
    // return 0;

    // for(auto i = clusterDistr.begin(); i != clusterDistr.end(); i++) 
    // { 
    //     cout << i->first << '\t' << i->second << '\n'; 
    // } 

    sortMap(clusterDistr);
   
    file.close();

	return 0;
}
