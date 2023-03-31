#!/bin/bash

fov="1 1.3 1.6"
for f in $fov
do
	python3 test.py -fov $f --gpu
done

