python ./6f_outside2_turn.py 0 1 --gpu
sleep 0.1
curl http://192.168.50.153:3074/movestop # 直走0
#curl http://192.168.50.153:3074/setaction?order=f:1 #橫移1
