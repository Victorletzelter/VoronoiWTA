#!/bin/bash

# Go in the results directory
cd ${MY_HOME}/results

# Delete the saved_zador directory if it exists
if [ -d "saved_zador" ]; then
    rm -r saved_zador
fi

mkdir saved_zador

cd ${MY_HOME}/scripts

python zador_computation.py