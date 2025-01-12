#!/bin/bash
#PJM -g "jh240057o"
#PJM -L "rscgrp=debug-o"
#PJM -L "node=4"
#PJM --mpi "proc=192"
#PJM -L "elapse=30:00"
#PJM -o "output.out"
#PJM -j
mpirun ./solver data/ss.mtx