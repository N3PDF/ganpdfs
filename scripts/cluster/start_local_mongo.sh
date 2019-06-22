#!/bin/bash

IP=`hostname --ip-address`
PORT=40000
echo $IP:$PORT
mongod --dbpath ./db --bind_ip $IP --port $PORT --noprealloc
