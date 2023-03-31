#!/bin/bash

name='test/social_stgcnn_FoV360'
for f in $*
do
	echo python3 test.py --gpu --wnadb -fov $f -wdn $name
done
