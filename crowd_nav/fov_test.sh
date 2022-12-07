#!/bin/bash

name = 'test/social_stgcnn_FoV234' 
for f in $*
do
	python3 test.py -fov $f --wandb -wdn $name --gpu
done
