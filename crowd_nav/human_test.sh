#!/bin/bash

name='test/social_stgcnn_fov360'
fov=.7 
for f in $*
do
	python3 test.py -fov $fov --wandb -wdn $name --gpu --human_num $f
done
