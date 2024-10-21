#!/bin/bash
#PJM -g "jh240057o"
#PJM -L "rscgrp=debug-o"
#PJM -L "node=16"
#PJM --mpi "proc=64"
#PJM -L "elapse=60:00"
mpirun ./solver data/transport.mtx bicgstab