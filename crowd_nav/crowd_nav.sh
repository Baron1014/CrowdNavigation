#source ./env/bin/activate
python ./6f_outside2.py 0 7 --gpu
sleep 0.1
curl http://192.168.50.153:3074/setaction?order=m:0 # 直走0
sleep 0.1
curl http://192.168.50.153:3074/setaction?order=t:$1 # 轉$度
sleep 3
curl http://192.168.50.153:3074/setaction?order=f:1 #橫移1
