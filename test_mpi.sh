#!/usr/bin/env bash

make &>/dev/null
if [ $? -ne 0 ]; then
    echo "Compilation failed"
    exit 1
fi

echo "Threads, epsilon, iterations, seconds, microseconds filename"
for i in {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}; do
    echo thread $i;
      for j in $(seq 5); do
          mpirun -np 5 --map-by node --hostfile 4hosts jacobimpi $i 0.0001 2000 2000 ./testMatrixes/ones2000.mtx;
      done
done
