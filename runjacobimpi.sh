mpirun -np 4 --map-by node --hostfile 4hosts --output-filename output/outjacobimpi --tag-output jacobimpi 6 .0001 500 500 ./testMatrixes/ones500.mtx
