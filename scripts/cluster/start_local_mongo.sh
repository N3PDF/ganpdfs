#!/bin/bash

IP=`hostname --ip-address`
PORT=121817
echo $IP:$PORT
mongod --dbpath ./db --bind_ip $IP --port $PORT --noprealloc
