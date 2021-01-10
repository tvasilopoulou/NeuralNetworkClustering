#include <iostream>
#include <fstream>
#include <string.h>
#include <algorithm> 
#include <list> 
#include <vector>
#include <bits/stdc++.h> 


/*./search -d train-images.idx3-ubyte -q t10k-images.idx3-ubyte -l1 train-labels.idx1-ubyte -l2 t10k-labels.idx1-ubyte  -o output.txt -EMD */

using namespace std;

int main(int argc, char * argv[]){
	// while(1){
	if (argc != 12)
	{
		cout << ("Please try running Search again. Number of arguments was different than expected.\n");
		return -1;
	}

    //check arguements in pairs
	if ((strcmp(argv[1],"-d") && strcmp(argv[3],"-d") && strcmp(argv[5],"-d") && strcmp(argv[7],"-d") && strcmp(argv[9],"-d"))
	|| (strcmp(argv[1],"-q") && strcmp(argv[3],"-q") && strcmp(argv[5],"-q") && strcmp(argv[7],"-q") && strcmp(argv[9],"-q"))
	|| (strcmp(argv[1],"-l1") && strcmp(argv[3],"-l1") && strcmp(argv[5],"-l1") && strcmp(argv[7],"-l1") && strcmp(argv[9],"-l1"))
	|| (strcmp(argv[1],"-l2") && strcmp(argv[3],"-l2") && strcmp(argv[5],"-l2") && strcmp(argv[7],"-l2") && strcmp(argv[9],"-l2"))
	|| (strcmp(argv[1],"-o") && strcmp(argv[3],"-o") && strcmp(argv[5],"-o") && strcmp(argv[7],"-o") && strcmp(argv[9],"-o"))
	|| (strcmp(argv[1],"-EMD") && strcmp(argv[3],"-EMD") && strcmp(argv[5],"-EMD") && strcmp(argv[7],"-EMD") && strcmp(argv[9],"-EMD") && strcmp(argv[11],"-EMD") ))
	{
		cout << ("Please try running Search again. Arguments given were either in the wrong order, or incorrect.\n");
		return -2;
	}

	char * inputFileOGS, * queryFileOGS, * inputLabels, *queryLabels, * outputFile;



/*-d*/
	//set variables assigned from input
	if (!(strcmp(argv[1],"-d"))){inputFileOGS=(char *)malloc(sizeof(argv[2]+1)) ; strncpy(inputFileOGS, argv[2], strlen(argv[2])+1);} 
	else if (!(strcmp(argv[3],"-d"))) { inputFileOGS=(char *)malloc(sizeof(argv[4]+1)) ; strncpy(inputFileOGS, argv[4], strlen(argv[4])+1); } 
	else if (!(strcmp(argv[5],"-d"))) { inputFileOGS=(char *)malloc(sizeof(argv[6]+1)) ; strncpy(inputFileOGS, argv[6], strlen(argv[6])+1); } 
	else if (argv[7]!=NULL && !(strcmp(argv[7],"-d"))) { inputFileOGS=(char *)malloc(sizeof(argv[8]+1)) ; strncpy(inputFileOGS, argv[8], strlen(argv[8])+1); } 		
	else if (argv[9]!=NULL && !(strcmp(argv[9],"-d"))) { inputFileOGS=(char *)malloc(sizeof(argv[10]+1)) ; strncpy(inputFileOGS, argv[10], strlen(argv[10])+1); } 
	


/*-q*/
	if (!(strcmp(argv[1],"-q"))){queryFileOGS=(char *)malloc(sizeof(argv[2]+1)) ; strncpy(queryFileOGS, argv[2], strlen(argv[2])+1);} 
	else if (!(strcmp(argv[3],"-q"))) { queryFileOGS=(char *)malloc(sizeof(argv[4]+1)) ; strncpy(queryFileOGS, argv[4], strlen(argv[4])+1); } 
	else if (!(strcmp(argv[5],"-q"))) { queryFileOGS=(char *)malloc(sizeof(argv[6]+1)) ; strncpy(queryFileOGS, argv[6], strlen(argv[6])+1); } 
	else if (argv[7]!=NULL && !(strcmp(argv[7],"-q"))) { queryFileOGS=(char *)malloc(sizeof(argv[8]+1)) ; strncpy(queryFileOGS, argv[8], strlen(argv[8])+1); } 		
	else if (argv[9]!=NULL && !(strcmp(argv[9],"-q"))) { queryFileOGS=(char *)malloc(sizeof(argv[10]+1)) ; strncpy(queryFileOGS, argv[10], strlen(argv[10])+1); } 


/*-l1*/	
	if (!(strcmp(argv[1],"-l1"))){inputLabels=(char *)malloc(sizeof(argv[2]+1)) ; strncpy(inputLabels, argv[2], strlen(argv[2])+1);} 
	else if (!(strcmp(argv[3],"-l1"))) { inputLabels=(char *)malloc(sizeof(argv[4]+1)) ; strncpy(inputLabels, argv[4], strlen(argv[4])+1); } 
	else if (!(strcmp(argv[5],"-l1"))) { inputLabels=(char *)malloc(sizeof(argv[6]+1)) ; strncpy(inputLabels, argv[6], strlen(argv[6])+1); } 
	else if (!(strcmp(argv[7],"-l1"))) { inputLabels=(char *)malloc(sizeof(argv[8]+1)) ; strncpy(inputLabels, argv[8], strlen(argv[8])+1); } 		
	else if (argv[9]!=NULL && !(strcmp(argv[9],"-l1"))) { inputLabels=(char *)malloc(sizeof(argv[10]+1)) ; strncpy(inputLabels, argv[10], strlen(argv[10])+1); } 
	


/*-q*/
	//set variables assigned from input
	if (!(strcmp(argv[1],"-l2"))){queryLabels=(char *)malloc(sizeof(argv[2]+1)) ; strncpy(queryLabels, argv[2], strlen(argv[2])+1);} 
	else if (!(strcmp(argv[3],"-l2"))) { queryLabels=(char *)malloc(sizeof(argv[4]+1)) ; strncpy(queryLabels, argv[4], strlen(argv[4])+1); } 
	else if (!(strcmp(argv[5],"-l2"))) { queryLabels=(char *)malloc(sizeof(argv[6]+1)) ; strncpy(queryLabels, argv[6], strlen(argv[6])+1); } 
	else if (!(strcmp(argv[7],"-l2"))) { queryLabels=(char *)malloc(sizeof(argv[8]+1)) ; strncpy(queryLabels, argv[8], strlen(argv[8])+1); } 		
	else if (argv[9]!=NULL && !(strcmp(argv[9],"-l2"))) { queryLabels=(char *)malloc(sizeof(argv[10]+1)) ; strncpy(queryLabels, argv[10], strlen(argv[10])+1); } 
	


/*-o*/
	//set variables assigned from input
	if (!(strcmp(argv[1],"-o"))){outputFile=(char *)malloc(sizeof(argv[2]+1)) ; strncpy(outputFile, argv[2], strlen(argv[2])+1);} 
	else if (!(strcmp(argv[3],"-o"))) { outputFile=(char *)malloc(sizeof(argv[4]+1)) ; strncpy(outputFile, argv[4], strlen(argv[4])+1); } 
	else if (!(strcmp(argv[5],"-o"))) { outputFile=(char *)malloc(sizeof(argv[6]+1)) ; strncpy(outputFile, argv[6], strlen(argv[6])+1); } 
	else if (!(strcmp(argv[7],"-o"))) { outputFile=(char *)malloc(sizeof(argv[8]+1)) ; strncpy(outputFile, argv[8], strlen(argv[8])+1); } 		
	else if (argv[9]!=NULL && !(strcmp(argv[9],"-o"))) { outputFile=(char *)malloc(sizeof(argv[10]+1)) ; strncpy(outputFile, argv[10], strlen(argv[10])+1); } 
	


	/*END OF ARGUMENT CHECK*/
	/*----------------------------------------------------------------------------------------------------------------------------------*/

	printf("%s %s %s %s %s\n", inputFileOGS, queryFileOGS, inputLabels, queryLabels, outputFile);
	string command = "python search.py " + inputFileOGS + " " + queryFileOGS + " "  + inputLabels + " " + queryLabels + "" + outputFile;
	system(command.c_str());


	return 0;
}