#!/bin/bash
#PJM -L "rscunit=fx"
#PJM -L "rscgrp=fx-debug"
#PJM -L "node=1"
#PJM --mpi "proc=48"
#PJM -L "elapse=60:00"
mpirun ./solver data/atmosmodd.mtx bicgstab