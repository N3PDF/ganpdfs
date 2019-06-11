#!/bin/bash

IP=hostname --ip-address
PORT=121817
workdir=`pwd`
echo $IP:$PORT
hyperopt-mongo-worker --mongo=$IP:$PORT/xganpdfs --workdir=$workdir
