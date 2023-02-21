#!/bin/bash

name='test/cadrl_orienv' 
for f in $*
do
	python3 test.py -fov $f --wandb -wdn $name 
done
