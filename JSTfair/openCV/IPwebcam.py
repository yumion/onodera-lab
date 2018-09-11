import urllib.request
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

# Replace the URL with your own IPwebcam shot.jpg IP:port
url='http://192.168.100.66:8080/shot.jpg'

# 黒色のマスク
def black_detect(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_min = np.array([0, 0, 0])
    hsv_max = np.array([179, 128, 100])
    mask = cv2.inRange(hsv, hsv_min, hsv_max)
    return mask

while True:

    # Use urllib to get the image and convert into a cv2 usable format
    imgResp=urllib.request.urlopen(url)
    imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
    img=cv2.imdecode(imgNp,-1)

    mask = black_detect(img)
    # ピクセル数を計算
    pix_area = cv2.countNonZero(mask)
    # パーセントを算出
    h, w = mask.shape # frameの面積
    per = round(100*float(pix_area)/(w * h),1)
    cv2.putText(mask, "Moment[px]: " + str(pix_area), (0, 50), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 5, cv2.LINE_AA)
    cv2.putText(mask, "Percent[%]: " + str(per), (0, 120), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 5, cv2.LINE_AA)

    # put the image on screen
    cv2.imshow('IPWebcam', mask)

    #To give the processor some less stress
    #time.sleep(0.1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
