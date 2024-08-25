#!/bin/bash
#PJM -L "rscunit=fx"
#PJM -L "rscgrp=fx-debug"
#PJM -L "node=12"
#PJM --mpi "proc=576"
#PJM -L "elapse=1:00"
mpirun -np 12 ./spmv data/Transport.mtx