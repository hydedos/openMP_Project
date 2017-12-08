#!/usr/bin/env bash

make &>/dev/null
if [ $? -ne 0 ]; then
    echo "Compilation failed"
    exit 1
fi

echo "Threads, epsilon, iterations, seconds, microseconds filename"
for i in {10,50,100,500,1000,2000,4000,8000,16000}; do
    echo ones$i.mtx;
      for j in $(seq 5); do
          mpirun -np 5 --hostfile 4hosts jacobimpi 6 0.0001 $i $i ./testMatrixes/ones$i.mtx;
      done
done
