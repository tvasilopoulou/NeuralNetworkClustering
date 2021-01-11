#
# In order to execute this "Makefile" just type "make"
#

OBJS 	= search.o genericFuncs.o classFuncs.o
SOURCE	= search.c genericFuncs.cpp classFuncs.cpp
HEADER  = header.hpp funcHeader.hpp
OUT  	= search 
CC	= g++
FLAGS   = -c #-g -Wall
# flags=$(shell gpgme-config --libs --cflags)
#-g -c -pedantic -ansi  -Wall
# -g option enables debugging mode
# -c flag generates object code for separate files

$(OUT): $(OBJS)
	$(CC) -g $(OBJS) -o $@

# create/compile the individual files >>separately<< 
search.o: search.cpp
	$(CC) $(FLAGS) search.cpp

genericFuncs.o: genericFuncs.cpp
	$(CC) $(FLAGS) genericFuncs.cpp

classFuncs.o: classFuncs.cpp
	$(CC) $(FLAGS) classFuncs.cpp


# clean house
clean:
	rm -f $(OBJS) $(OUT) gpgme*
