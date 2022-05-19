#!/bin/bash

config=$1
prefix=$2

export PATH="/media/compute/homes/pkenneweg/anaconda3/envs/test/bin:$PATH"
python3 ${prefix}/main.py --config ${prefix}/${config}
