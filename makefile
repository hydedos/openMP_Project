all: openmp

openmp: openmp.c
	gcc openmp.c -O3 -Wall -g -o openmp

.PHONY: clean
clean:
	rm -f openmp
