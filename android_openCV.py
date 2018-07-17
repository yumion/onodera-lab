#coding: utf-8
import androidhelper as android
import cv
import json
import time
import numpy as np
import os


# 赤色のマスク
def red_detect(img):
    # HSV色空間に変換
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # 赤色のHSVの値域1
    hsv_min = np.array([0, 127, 0])
    hsv_max = np.array([30, 255, 255])
    mask1 = cv.inRange(hsv, hsv_min, hsv_max)
    # 赤色のHSVの値域2
    hsv_min = np.array([150, 127, 0])
    hsv_max = np.array([179, 255, 255])
    mask2 = cv.inRange(hsv, hsv_min, hsv_max)

    return mask1 + mask2

# 面積計算
def calc_area(img):
    pix_area = cv.countNonZero(img)  # ピクセル数
    # パーセントを算出
    h, w = img.shape  # frameの面積
    per = round(100 * float(pix_area) / (w * h), 3)  # 比率

    return pix_area, per

# 重心を求める
def calc_center(img):
    mu = cv.moments(img, False)
    x, y = int(mu["m10"] / mu["m00"]), int(mu["m01"] / mu["m00"])
    # 重心を丸でくくる
    cv.circle(img, (x, y), 4, 100, 2, 4)

    return x, y

#ArduinoとAndroidのシリアル通信
def serialStart():
    enable = droid.usbserialHostEnable()

    l = droid.usbserialGetDeviceList().result.items()
    tk = str(l).split(',')[-1]
    h = tk.split(chr(34))[1]

    ret = droid.usbserialConnect(str(h))
    uuid = str(ret.result.split(chr(34))[-2])
    print('uuid: ', uuid)

    time.sleep(3)

    active = droid.usbserialActiveConnections()
    print('active: ', active)

    return uuid

#androidhelper起動
droid = android.Android()

path = '/storage/7E9B-5A00/Picture/'

#serial通信オープン
uuid = serialStart()


cnt = 0
while True:
    # 写真を取る
    droid.cameraCapturePicture(path + str(cnt) + 'image.png')
    # openCVで開く
    img = cv.imread(path + str(cnt) + 'image.png')
    mask = red_detect(img)  # 赤色検出
    cv.imwrite(path + str(cnt) + 'Redmask.png', mask)  # 保存

    # 面積を求める
    pix_area, per = calc_area(mask)
    # 重心を求める
    x, y = calc_center(mask)
    print("G({},{})".format(x, y))

    # 差分を取る
    if cnt >= 1:
        # 1つ前の画像の計算
        img0 = cv.imread(path + str(cnt - 1) + 'image.png')
        mask0 = red_detect(img0)
        _, per0 = calc_area(mask0)
        x0, y0 = calc_center(mask0)
        print("G0({},{})".format(x0, y0))
        # 差分
        diff_area = abs(per - per0)
        diff_center = np.array([x - x0, y - y0])
        print('diff_area: {0:.3f}'.format(diff_area))
        print('diff_center({})'.format(diff_center))

    if os.path.isfile(path + 'image.png') is True:  # 写真が取れたら前進
        # arduinoにシリアル書き込み #UTF-8:1->ASCII:49
        droid.usbserialWrite((u'1'.encode('utf-8')), uuid)

    if cnt == 1:  # cnt=2でリセットする
        cnt = 0
    else:
        cnt = cnt + 1
