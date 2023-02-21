#!/bin/bash

name='test/sstgcnn_fov234_orienv_human'
for f in $*
do
	python3 test.py -fov $f --wandb -wdn $name --gpu --human_num 8
	python3 test.py -fov $f --wandb -wdn $name --gpu --human_num 10
done
