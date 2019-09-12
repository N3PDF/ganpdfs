#!/bin/bash

IP=hostname --ip-address
PORT=40000
workdir=`pwd`
echo $IP:$PORT
hyperopt-mongo-worker --mongo=$IP:$PORT/wganpdfs --workdir=$workdir
