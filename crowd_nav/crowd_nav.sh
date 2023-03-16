#source ./env/bin/activate

python ./6f_outside1.py 0 17 --gpu
sleep 0.1
curl http://192.168.50.153:3074/setaction?order=m:0
sleep 0.1
curl http://192.168.50.153:3074/setaction?order=t:-34
sleep 13
python ./6f_outside2.py 7.5 17.45 --gpu
sleep 0.1
curl http://192.168.50.153:3074/setaction?order=m:0
curl http://192.168.50.153:3074/setaction?order=t:-34
sleep 3
curl http://192.168.50.153:3074/setaction?order=f:1
