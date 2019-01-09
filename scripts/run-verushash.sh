#!/bin/sh
PoolHost=na.luckpool.net
Port=3956
PublicVerusCoinAddress=RVjvbZuqSGLGDm1B9BFkbHWySPKEx4tfjQ
WorkerName=Worker1
#set working directory to the location of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR
./ccminer --a verus -o stratum+tcp://"${PoolHost}":"${Port}" -u "${PublicVerusCoinAddress}"."${WorkerName}"  "$@"
