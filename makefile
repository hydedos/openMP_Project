all: openmp

openmp: openmp.c
	mpicc -fopenmp openmp.c -O3 -Wall -g -o jacobimpi

.PHONY: clean
clean:
	rm -f openmp
