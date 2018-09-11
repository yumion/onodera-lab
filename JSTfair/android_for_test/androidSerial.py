#coding: utf-8
import androidhelper as android
import json
import time

droid = android.Android()
enable = droid.usbserialHostEnable()
print("enable ",str(enable))

l = droid.usbserialGetDeviceList().result.items()

tk = str(l).split(",")[-1]

h = tk.split(chr(34))[1]

ret = droid.usbserialConnect(str(h))

uuid = str(ret.result.split(chr(34))[-2])

time.sleep(2)

active = droid.usbserialActiveConnections()
print("active :",active)
time.sleep(5)

for i in range(5):
    time.sleep(1)
    print(droid.usbserialRead(uuid))
    droid.usbserialWrite((u'53'.encode('utf-8')), uuid) #asciiで出力される
    
for i in range(5):
    time.sleep(1)
    droid.usbserialWrite('49', uuid)
    
for i in range(5):
    time.sleep(1)
    droid.usbserialWrite('60', uuid)


for i in range(5):
    time.sleep(1)
    #print(droid.usbserialRead(uuid))
    droid.usbserialWrite((u'53'.encode('utf-8')), uuid)
    
