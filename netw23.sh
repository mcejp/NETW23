#!/bin/sh

../100worlds/settlements/websocketd \
    --address=localhost \
    --port=4000 \
    --staticdir=../100worlds/settlements/static \
    env PYTHONPATH=`pwd`/../100worlds/settlements/settlements python -u main.py
